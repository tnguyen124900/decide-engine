# Generates tests/reference/netmeta_*.json - reference results from the R
# package netmeta, used by tests/test_nma_reference.py to check DECIDE's
# frequentist NMA engine (effects, tau2, Q, prediction intervals,
# P-scores, node-splitting, design-by-treatment decomposition and the
# contribution matrix) on netmeta's own example data sets.
#
# Run once from the heor-tree-engine folder (R 4.x, netmeta >= 2.8):
#
#   Rscript tests/reference/make_netmeta_reference.R
#
# Requires: install.packages(c("netmeta", "jsonlite"))
# Then:     python -m pytest tests/test_nma_reference.py -v

suppressPackageStartupMessages({
  library(netmeta)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- helpers ----------------------------------------------------------------

named_effects <- function(te, se, reference) {
  trts <- setdiff(rownames(te), reference)
  setNames(lapply(trts, function(t) list(TE = te[t, reference], seTE = se[t, reference])), trts)
}

matrix_to_list <- function(m) {
  # {"row": {"col": value}}, NA -> null
  if (is.null(m)) return(NULL)
  rows <- rownames(m); cols <- colnames(m)
  setNames(lapply(rows, function(r) setNames(lapply(cols, function(c) {
    v <- m[r, c]; if (is.na(v)) NULL else unname(v)
  }), cols)), rows)
}

split_frame <- function(sp, which) {
  d <- sp[[which]]
  if (is.null(d)) return(NULL)
  lapply(seq_len(nrow(d)), function(i) {
    row <- list(comparison = as.character(d$comparison[i]))
    for (col in intersect(c("TE", "seTE", "statistic", "p"), names(d))) row[[col]] <- unname(d[[col]][i])
    row
  })
}

extras <- function(nm) {
  rk <- tryCatch(netrank(nm, small.values = "desirable"), error = function(e) NULL)
  sp <- tryCatch(netsplit(nm), error = function(e) NULL)
  dd <- tryCatch(decomp.design(nm), error = function(e) NULL)
  ct <- tryCatch(netcontrib(nm, method = "shortestpath"), error = function(e) NULL)
  list(
    k = nm$k, n = nm$n, m = nm$m,
    tau2 = nm$tau2, tau = nm$tau, Q = nm$Q, df = nm$df.Q, pval_Q = nm$pval.Q, I2 = nm$I2,
    prediction = list(
      level = nm$level.predict,
      lower = matrix_to_list(nm$lower.predict), upper = matrix_to_list(nm$upper.predict),
      seTE = matrix_to_list(nm$seTE.predict)
    ),
    pscore = if (is.null(rk)) NULL else list(common = as.list(rk$ranking.common), random = as.list(rk$ranking.random)),
    netsplit = if (is.null(sp)) NULL else list(
      direct_common = split_frame(sp, "direct.common"), indirect_common = split_frame(sp, "indirect.common"),
      compare_common = split_frame(sp, "compare.common"),
      direct_random = split_frame(sp, "direct.random"), indirect_random = split_frame(sp, "indirect.random"),
      compare_random = split_frame(sp, "compare.random")
    ),
    decomp = if (is.null(dd)) NULL else list(
      Q_decomp = lapply(rownames(dd$Q.decomp), function(r) list(
        what = r, Q = dd$Q.decomp[r, "Q"], df = dd$Q.decomp[r, "df"], pval = dd$Q.decomp[r, "pval"])),
      Q_inc_random = list(Q = dd$Q.inc.random[1, "Q"], df = dd$Q.inc.random[1, "df"],
                          pval = dd$Q.inc.random[1, "pval"],
                          tau2_within = if (!is.null(dd$tau2.within)) dd$tau2.within
                                        else if (!is.null(dd$tau.within)) dd$tau.within^2 else NULL),
      names = names(dd)
    ),
    contribution = if (is.null(ct)) NULL else list(common = matrix_to_list(ct$common), random = matrix_to_list(ct$random))
  )
}

write_reference <- function(name, payload) {
  write_json(payload, file.path(out_dir, paste0("netmeta_", name, ".json")),
             auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null")
  cat("Wrote", name, "\n")
}

arm_studies <- function(data, arms_cols, fields) {
  # fields: named list DECIDE name -> data column prefix, e.g. list(events = "event", n = "n")
  lapply(seq_len(nrow(data)), function(r) {
    arms <- list()
    for (i in seq_len(arms_cols)) {
      t <- data[[paste0("treat", i)]][r]
      if (!is.na(t) && nzchar(as.character(t))) {  # smokingcessation uses "" for a missing 3rd arm
        arm <- list(treatment = as.character(t))
        for (f in names(fields)) arm[[f]] <- data[[paste0(fields[[f]], i)]][r]
        arms[[length(arms) + 1]] <- arm
      }
    }
    list(id = as.character(data$studlab[r]), arms = arms)
  })
}

export <- function(nm, data, arms_cols, fields, name, outcome, measure, reference) {
  write_reference(name, c(list(
    name = name, outcome = outcome, measure = measure, reference = reference,
    netmeta_version = as.character(packageVersion("netmeta")),
    studies = arm_studies(data, arms_cols, fields),
    fixed = named_effects(nm$TE.common, nm$seTE.common, reference),
    random = named_effects(nm$TE.random, nm$seTE.random, reference)
  ), extras(nm)))
}

cols <- function(data, prefix, k) lapply(seq_len(k), function(i) data[[paste0(prefix, i)]])
fit <- function(p, reference) netmeta(p, common = TRUE, random = TRUE, reference.group = reference,
                                       method.tau = "DL", prediction = TRUE, details.chkmultiarm = FALSE)

# ---- binary (events / n): OR, RR, RD ---------------------------------------
data(smokingcessation)                       # 24 studies, 4 treatments, 2 three-arm studies
smk <- smokingcessation
smk$studlab <- paste0("S", seq_len(nrow(smk)))
safely <- function(label, expr) tryCatch(expr, error = function(e) cat("ERROR in", label, ":", conditionMessage(e), "\n"))

# RR / RD: netmeta's continuity correction for zero cells differs from DECIDE's
# (DECIDE adds 0.5 to every cell of the study, as for OR), so those two
# checks use the studies without zero cells only.
zero <- with(smk, (!is.na(event1) & event1 == 0) | (!is.na(event2) & event2 == 0) | (!is.na(event3) & event3 == 0))
for (sm in c("OR", "RR", "RD")) safely(sm, {
  d <- if (sm == "OR") smk else smk[!zero, ]
  p <- pairwise(treat = cols(d, "treat", 3), event = cols(d, "event", 3), n = cols(d, "n", 3),
                studlab = d$studlab, sm = sm)
  export(fit(p, "A"), d, 3, list(events = "event", n = "n"),
         if (sm == "OR") "smokingcessation" else paste0("smokingcessation_", tolower(sm)), "binary", sm, "A")
})

# ---- continuous (mean, sd, n): MD, SMD --------------------------------------
data(Franchini2012)                          # 6 studies, 3 three-arm
safely("Franchini2012", {
fr <- Franchini2012
fr$studlab <- as.character(fr$Study)
names(fr) <- sub("^Treatment", "treat", names(fr))
names(fr) <- sub("^y", "mean", names(fr))
p <- pairwise(treat = cols(fr, "treat", 3), mean = cols(fr, "mean", 3), sd = cols(fr, "sd", 3),
              n = cols(fr, "n", 3), studlab = fr$studlab, sm = "MD")
export(fit(p, "Placebo"), fr, 3, list(mean = "mean", sd = "sd", n = "n"), "franchini_md", "continuous", "MD", "Placebo")
# SMD: netmeta pools the SD per pair of arms, DECIDE pools it over all arms of
# a study, so the SMD check uses the two-arm studies only.
fr2 <- fr[is.na(fr$treat3), ]
p <- pairwise(treat = cols(fr2, "treat", 2), mean = cols(fr2, "mean", 2), sd = cols(fr2, "sd", 2),
              n = cols(fr2, "n", 2), studlab = fr2$studlab, sm = "SMD")
export(fit(p, "Placebo"), fr2, 2, list(mean = "mean", sd = "sd", n = "n"), "franchini_smd_twoarm", "continuous", "SMD", "Placebo")
})

# ---- rates (events / person-time): IRR -------------------------------------
data(dietaryfat)                             # 10 studies, 3 treatments, 1 three-arm
safely("dietaryfat", {
dfat <- dietaryfat
dfat$studlab <- paste0("D", seq_len(nrow(dfat)))
p <- pairwise(treat = cols(dfat, "treat", 3), event = cols(dfat, "d", 3), time = cols(dfat, "years", 3),
              studlab = dfat$studlab, sm = "IRR")
export(fit(p, "1"), dfat, 3, list(events = "d", exposure = "years"), "dietaryfat", "rate", "RATE_RATIO", "1")
})

cat("Done. Now run: python -m pytest tests/test_nma_reference.py -v\n")
