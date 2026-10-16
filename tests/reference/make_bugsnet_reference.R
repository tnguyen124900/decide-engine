# Generates tests/reference/bugsnet_*.json - a second independent Bayesian NMA
# reference, from the R package BUGSnet, to cross-check the
# same posterior the gemtc reference checks -- two independent MCMC
# implementations agreeing with DECIDE's slice-sampler within Monte-Carlo
# tolerance is much stronger evidence than either alone.
#
# Run once from the heor-tree-engine folder (R 4.x, BUGSnet, JAGS installed):
#
#   Rscript tests/reference/make_bugsnet_reference.R
#
# Requires: JAGS (brew install jags), then
#           install.packages(c("BUGSnet", "jsonlite"))
# Then:     python -m pytest tests/test_nma_bayes_reference.py -v

suppressPackageStartupMessages({
  library(BUGSnet)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(smokingcessation, package = "netmeta")
smk <- smokingcessation
smk$studlab <- paste0("S", seq_len(nrow(smk)))

arms <- do.call(rbind, lapply(seq_len(nrow(smk)), function(r) {
  do.call(rbind, lapply(1:3, function(i) {
    t <- smk[[paste0("treat", i)]][r]
    if (is.na(t) || !nzchar(as.character(t))) return(NULL)
    data.frame(study = smk$studlab[r], treatment = as.character(t),
               r = smk[[paste0("event", i)]][r], n = smk[[paste0("n", i)]][r],
               stringsAsFactors = FALSE)
  }))
}))

# BUGSnet random-effects binomial/logit.
prep <- data.prep(arm.data = arms, varname.t = "treatment", varname.s = "study")
model <- nma.model(data = prep, outcome = "r", N = "n", reference = "A",
                   family = "binomial", link = "logit", effects = "random")
fit <- nma.run(model, n.adapt = 5000, n.iter = 20000, n.burnin = 5000,
               n.chains = 4, DIC = TRUE)

# Posterior summary per d.<treat>.A and sd (tau).
s <- fit$samples
sm <- summary(s)
stats <- sm$statistics
quants <- sm$quantiles
param_names <- rownames(stats)

effects <- setNames(lapply(param_names[grepl("^d\\.", param_names)], function(p) {
  list(mean = unname(stats[p, "Mean"]), sd = unname(stats[p, "SD"]),
       lower = unname(quants[p, "2.5%"]), upper = unname(quants[p, "97.5%"]))
}), param_names[grepl("^d\\.", param_names)])

tau_row <- param_names[grepl("^sd", param_names)][1]
tau <- if (!is.na(tau_row)) list(mean = unname(stats[tau_row, "Mean"]),
                                 sd = unname(stats[tau_row, "SD"]),
                                 lower = unname(quants[tau_row, "2.5%"]),
                                 upper = unname(quants[tau_row, "97.5%"])) else NULL

write_json(list(
  name = "smokingcessation", outcome = "binary", measure = "OR",
  reference = "A", link = "logit", likelihood = "binomial",
  bugsnet_version = as.character(packageVersion("BUGSnet")),
  effects = effects, tau = tau
), file.path(out_dir, "bugsnet_smokingcessation.json"),
  auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null")
cat("Wrote bugsnet_smokingcessation\n")
