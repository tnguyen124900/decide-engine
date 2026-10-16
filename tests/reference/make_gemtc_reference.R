# Generates tests/reference/gemtc_*.json - Bayesian NMA reference results from
# the R package gemtc, used by tests/test_nma_bayes_reference.py to check
# DECIDE's Bayesian NMA engine (posterior mean/sd per treatment effect, tau,
# rank probabilities / SUCRA) against the de-facto standard Cochrane/HTA
# Bayesian NMA package, on the same smokingcessation data the frequentist
# netmeta reference uses.
#
# Run once from the heor-tree-engine folder (R 4.x, gemtc, JAGS installed):
#
#   Rscript tests/reference/make_gemtc_reference.R
#
# Requires: JAGS (brew install jags), then
#           install.packages(c("gemtc", "jsonlite"))
# Then:     python -m pytest tests/test_nma_bayes_reference.py -v

suppressPackageStartupMessages({
  library(gemtc)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_reference <- function(name, payload) {
  write_json(payload, file.path(out_dir, paste0("gemtc_", name, ".json")),
             auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null")
  cat("Wrote", name, "\n")
}

# smokingcessation, one row per arm (gemtc's arm-level format).
data(smokingcessation, package = "netmeta")
smk <- smokingcessation
smk$studlab <- paste0("S", seq_len(nrow(smk)))

# Long arm-level frame gemtc wants: study / treatment / responders / sampleSize.
arms <- do.call(rbind, lapply(seq_len(nrow(smk)), function(r) {
  do.call(rbind, lapply(1:3, function(i) {
    t <- smk[[paste0("treat", i)]][r]
    if (is.na(t) || !nzchar(as.character(t))) return(NULL)
    data.frame(study = smk$studlab[r], treatment = as.character(t),
               responders = smk[[paste0("event", i)]][r],
               sampleSize = smk[[paste0("n", i)]][r],
               stringsAsFactors = FALSE)
  }))
}))

network <- mtc.network(data.ab = arms, description = "Smoking cessation")

# Random-effects, log-odds-ratio, vague priors matching DECIDE's defaults
# (N(0, 100^2) on each d_k, Uniform(0, 5) on tau).
model <- mtc.model(network, type = "consistency", linearModel = "random",
                   link = "logit", likelihood = "binom",
                   om.scale = 5,  # upper bound of the uniform tau prior
                   n.chain = 4)
fit <- mtc.run(model, sampler = "JAGS", n.adapt = 5000, n.iter = 20000,
               thin = 1, n.burnin = 5000)

s <- summary(fit)
summ <- s$summaries
stats <- summ$statistics          # rows: d.B.A, ..., sd.d; cols: Mean, SD, ...
quants <- summ$quantiles

param_names <- rownames(stats)
effects <- setNames(lapply(param_names[grepl("^d\\.", param_names)], function(p) {
  list(mean = unname(stats[p, "Mean"]), sd = unname(stats[p, "SD"]),
       lower = unname(quants[p, "2.5%"]), upper = unname(quants[p, "97.5%"]))
}), param_names[grepl("^d\\.", param_names)])

tau_row <- if ("sd.d" %in% param_names) "sd.d" else param_names[grepl("^sd", param_names)][1]
tau <- if (!is.na(tau_row)) list(mean = unname(stats[tau_row, "Mean"]),
                                 sd = unname(stats[tau_row, "SD"]),
                                 lower = unname(quants[tau_row, "2.5%"]),
                                 upper = unname(quants[tau_row, "97.5%"])) else NULL

# Rank probabilities (rank 1 = best; smoking cessation is "larger event count
# is worse", but gemtc ranks on the analysis scale -- we record ranks as-is).
rk <- tryCatch(rank.probability(fit, preferredDirection = -1), error = function(e) NULL)
rank_probs <- if (is.null(rk)) NULL else as.list(as.data.frame(rk))

write_reference("smokingcessation", list(
  name = "smokingcessation", outcome = "binary", measure = "OR",
  reference = "A", link = "logit", likelihood = "binom",
  gemtc_version = as.character(packageVersion("gemtc")),
  settings = list(n_chain = 4, n_adapt = 5000, n_iter = 20000, n_burnin = 5000),
  studies = lapply(seq_len(nrow(smk)), function(r) {
    list(id = smk$studlab[r], arms = Filter(Negate(is.null), lapply(1:3, function(i) {
      t <- smk[[paste0("treat", i)]][r]
      if (is.na(t) || !nzchar(as.character(t))) return(NULL)
      list(treatment = as.character(t), events = smk[[paste0("event", i)]][r],
           n = smk[[paste0("n", i)]][r])
    })))
  }),
  effects = effects, tau = tau, rank_probabilities = rank_probs
))
