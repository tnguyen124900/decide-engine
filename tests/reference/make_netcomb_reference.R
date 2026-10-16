# Generates tests/reference/netcomb_additive.json: additive component NMA
# (netmeta::netcomb) on a small fixed data set, used by
# tests/test_nma_components_reference.py to check DECIDE's component NMA
# (nma_components.run_component_nma) against R.
#
# Run once from the heor-tree-engine folder:
#   Rscript tests/reference/make_netcomb_reference.R
# Requires: install.packages(c("netmeta", "jsonlite"))
# The same 9 contrasts are hard-coded in the Python test; keep them in sync.

suppressPackageStartupMessages({ library(netmeta); library(jsonlite) })

d <- data.frame(
  studlab = paste0("s", 1:9),
  treat1  = c("X",  "Y",  "Z",  "X+Y", "X+Z", "X+Y", "X+Z", "X+Y", "X"),
  treat2  = c("P",  "P",  "P",  "P",   "P",   "X",   "Z",   "Y",   "P"),
  TE      = c(0.5,  0.9,  0.3,  1.6,   0.9,   0.8,   0.5,   0.5,   1.4),
  seTE    = c(0.2,  0.25, 0.2,  0.3,   0.3,   0.25,  0.3,   0.25,  0.3),
  stringsAsFactors = FALSE
)
nm <- netmeta(TE, seTE, treat1, treat2, studlab, data = d, reference.group = "P", sm = "MD")
nc <- netcomb(nm, inactive = "P", sep.comps = "+")

pick <- function(obj, ...) { for (n in c(...)) if (!is.null(obj[[n]])) return(obj[[n]]); NULL }
as_named <- function(v) if (is.null(v)) NULL else as.list(v)
out <- list(
  components_random = as_named(pick(nc, "Comp.random")),
  se_components_random = as_named(pick(nc, "seComp.random")),
  components_common = as_named(pick(nc, "Comp.common", "Comp.fixed")),
  se_components_common = as_named(pick(nc, "seComp.common", "seComp.fixed")),
  tau2 = nc$tau2, Q_additive = nc$Q.additive, Q_standard = nc$Q.standard,
  Q_diff = nc$Q.diff, df_diff = nc$df.Q.diff, p_diff = nc$pval.Q.diff
)
write_json(out, file.path("tests", "reference", "netcomb_additive.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null")
cat("Wrote netcomb_additive.json. Now: python -m pytest tests/test_nma_components_reference.py -v\n")
