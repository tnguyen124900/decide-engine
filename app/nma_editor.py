"""The NMA Editor screen (model_type == "nma").

Layout follows the NMA Editor mockup (Design canvas "NMA Editor – Layout"):
sticky gradient toolbar (Change model / Auto arrange / Check network /
Run NMA / Inconsistency / Dashboard / Send to EE model), a sidebar with
File / Model settings / MCMC / Treatments, then Study data (arm-level or
contrast-level tables) beside the Network plot (2D or 3D, node colours by
class / single / custom), a network status banner, and the collapsible
Base case results + Model details panels below.

Kept out of app.py (which is edited by other work in parallel) - app.py's
render_nma_app() just calls render_nma_editor(T, ctx), passing the shared
helpers the screen needs through `ctx` instead of importing app.py.

Engine: heor_tree_engine.nma_* (frequentist for now; Bayesian and
node-splitting come next, so Inconsistency / Send to EE / MCMC are shown
but disabled).
"""

from __future__ import annotations

import html as html_lib
from dataclasses import replace as dc_replace
import json
import os
import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from contextlib import contextmanager


@contextmanager
def _nma_setting_panel(name: str, title: str):
    """NMA's sidebar sections. Browser: plain sidebar. Desktop app: the sidebar is gone and
    these open as a centred dialog from Setting > NMA settings."""
    if os.environ.get("DECIDE_DESKTOP") != "1":
        yield
        return
    is_open = st.session_state.get("_setting_open") == name
    cls = f"div.st-key-setting_{name}"
    css = f"{cls}{{display:none !important;}}"
    if is_open:
        css = f"""
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
            transform: none !important; will-change: auto !important; filter: none !important;
            contain: none !important; perspective: none !important; }}
        {cls} {{
            display: block !important; position: fixed !important; top: 50% !important; left: 50% !important;
            transform: translate(-50%, -50%) !important; width: min(760px, 88vw) !important;
            max-height: 86vh !important; overflow-y: auto !important; z-index: 1000002 !important; visibility: visible !important;
            background: #ffffff; border: 1px solid #1d4f9b; border-radius: 10px; padding: 20px 30px 26px 30px; box-sizing: border-box;
            box-shadow: 0 0 0 100vmax rgba(15, 30, 60, 0.45), 0 12px 40px rgba(0,0,0,0.35); }}
        {cls} [data-baseweb="select"] > div, {cls} [data-baseweb="input"], {cls} [data-baseweb="base-input"],
        {cls} [data-baseweb="textarea"], {cls} div[data-testid="stNumberInputContainer"] {{ background: #eceff4 !important; border-color: #cfd6e2 !important; }}
        {cls} [data-baseweb="select"] > div, {cls} input {{ color: #1f2a3a !important; }}
        {cls} input, {cls} textarea {{ background: #eceff4 !important; }}
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    with st.container(key=f"setting_{name}"):
        if is_open:
            head, close = st.columns([14, 1], vertical_alignment="center")
            head.markdown(f"### {title}")
            if close.button("✕", key=f"setting_{name}_close_btn", help="Close"):
                st.session_state._setting_open = None
                st.rerun()
        yield


from model_files import ModelFiles
from streamlit.components.v1 import html as st_html

import nma_editor_support as sup
from nma_table_component import nma_table
from heor_tree_engine.nma_freq import run_nma_frequentist
from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian
from heor_tree_engine.nma_bayes_exact import EXACT_LIKELIHOOD, LIKELIHOOD_LABELS, ExactResult, run_nma_bayesian_exact
from heor_tree_engine import nma_priors
from heor_tree_engine.nma_evidence import build_evidence_set, merge_into_document
from heor_tree_engine.nma_cinema import (CONCERN_LEVELS, CONFIDENCE, DOMAINS, REPORTING_LEVELS, cinema_report_rows,
                                         default_margin, normalize_level, run_cinema)
from heor_tree_engine.nma_absolute import ABSOLUTE_MEASURES, absolute_effects, observed_baseline_risk
from heor_tree_engine.nma_contribution import contribution_matrix
from heor_tree_engine import nma_components, nma_design, nma_km_reconstruct, nma_maic, nma_mbnma, nma_mlnmr, nma_survival, nma_threshold, nma_timevarying
from heor_tree_engine.nma_conversions import (combine_arms, mean_sd_from_median, sd_from_ci, sd_from_se, se_from_ci,
                                              se_from_p)
from heor_tree_engine.nma_funnel import comparison_adjusted_funnel
import nma_prisma
import nma_report
import nma_sof
from heor_tree_engine.nma_models import NMAResult
from heor_tree_engine.nma_regression import BASELINE_RISK, effect_curve, effects_at, run_meta_regression
from heor_tree_engine.nma_inconsistency import run_inconsistency
from heor_tree_engine.nma_sensitivity import exclude_studies, influence_summary, leave_one_out
from heor_tree_engine.nma_models import ARM_LEVEL_MEASURES, MEASURES, is_ratio_measure
from heor_tree_engine.nma_network import summarize_network
from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_validation import validate_nma

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "NMA Editor",
        "subtitle": "© 2026 Trang Nguyen",
        "auto_arrange": "Auto arrange (reset network view)",
        "check_network": "Check model",
        "run_nma": "Run NMA",
        "inconsistency": "Inconsistency",
        "inconsistency_help": "Node-splitting (direct vs indirect) and the design-by-treatment test",
        "incons_title": "Inconsistency",
        "incons_global": "Global test (design-by-treatment interaction)",
        "incons_row_total": "Total (consistency model)",
        "incons_row_within": "Within designs (heterogeneity)",
        "incons_row_between": "Between designs (inconsistency)",
        "incons_row_between_re": "Between designs, after allowing for heterogeneity (τ² within designs)",
        "incons_flag": "Evidence of inconsistency (p < 0.05) - check the flagged comparisons and study characteristics before relying on network estimates.",
        "incons_ok": "No statistically significant inconsistency (p ≥ 0.05). Low power with few loops - read alongside the node-split table.",
        "incons_split": "Node-splitting: direct vs indirect evidence",
        "incons_split_hint": "Rows in orange: p < 0.05. Ratio = indirect / direct (ratio measures) or difference = direct - indirect.",
        "incons_none": "No comparison has both direct and independent indirect evidence (no closed loops), so node-splitting is not possible.",
        "incons_not_split": "Only direct or only indirect evidence (not splittable)",
        "incons_designs": "Q by design",
        "prediction_note": "Light bars: {pct} prediction interval (random effects, t with {df} df).",
        "dashboard": "Dashboard",
        "send_to_ee": "Send to EE model",
        "send_to_ee_help": "Add the NMA relative effects (with PSA distributions and correlations) as parameters of a Markov / PSM / Tree / Microsimulation model",
        "ee_title": "Evidence Set → EE model",
        "ee_target": "Target model",
        "ee_anchor": "Versus (baseline arm of the EE model)",
        "ee_treatments": "Treatments to export",
        "ee_prefix": "Name prefix (optional)",
        "ee_add_psa": "Also add PSA distributions + correlations",
        "ee_preview": "Parameters that will be added or updated",
        "ee_corr": "{n} correlation(s) between these effects will be added to the model's PSA (they share the {anchor} evidence).",
        "ee_apply": "Add to {model}",
        "ee_done": "{model}: added {added}; updated {updated}. Open {model} → Parameters to use them.",
        "ee_formula": "How to apply in the model (DECIDE expression; p_base = baseline probability parameter of {anchor}):",
        "ee_download": "Download evidence set (JSON)",
        "ee_no_target": "That model has not been opened in this session yet - open it once, then come back.",
        "characteristics": "Study characteristics",
        "characteristics_help": "Optional, one row per study: used for the transitivity table (Check network) and the risk-of-bias sensitivity analysis (Dashboard) and CINeMA (More analyses). Risk of bias and Indirectness: Low / Some / High.",
        "chars_vars_title": "Variables (columns) of this table",
        "chars_vars_help": "Study, Risk of bias, Indirectness and Notes are fixed. Add, rename, retype or remove the other variables; numeric variables are summarised per comparison in the transitivity table and can be used as covariates in network meta-regression.",
        "chars_var_label": "Variable",
        "chars_var_type": "Type",
        "chars_type_number": "Number",
        "chars_type_text": "Text",
        "chars_var_new": "New variable",
        "chars_var_add": "Add variable",
        "chars_var_delete": "Remove '{name}' (its values are dropped)",
        "chars_var_reset": "Reset to default variables",
        "transitivity": "Transitivity: study characteristics by comparison",
        "transitivity_hint": "Mean (min–max) across the studies of each direct comparison. Comparisons that differ a lot in effect modifiers (age, severity, follow-up...) make indirect comparisons less trustworthy.",
        "transitivity_plot": "Transitivity plot: each study's value by comparison",
        "transitivity_plot_hint": "Each dot is one study, coloured by risk of bias; the bar spans the interquartile range with the median marked. Comparisons whose dots sit in different places differ in this potential effect modifier, which weakens the indirect comparisons built on them.",
        "transitivity_variable": "Variable",
        "transitivity_plot_empty": "No numeric study characteristics entered yet (Study data → Characteristics).",
        "transitivity_text_counts": "Text characteristics: studies per category and comparison",
        "rep_fig_transitivity": "Figure 1b. Study characteristics by comparison (transitivity): each dot is a study, coloured by risk of bias; the bar spans the interquartile range with the median marked.",
        "rob_title": "Sensitivity: excluding studies at high risk of bias",
        "rob_none": "No study is marked High risk of bias (Study data → Study characteristics).",
        "rob_excluded": "Excluded: {ids}",
        "more": "More analyses",
        "more_help": "CINeMA confidence rating, contribution matrix, network meta-regression and the comparison-adjusted funnel plot",
        "more_title": "More analyses",
        "nmr_tab": "Network meta-regression",
        "funnel_tab": "Funnel plot",
        "nmr_covariate": "Covariate",
        "nmr_baseline": "Baseline risk (reference arm, analysis scale)",
        "nmr_interaction": "Interaction",
        "nmr_shared": "Shared (one β for all vs reference)",
        "nmr_specific": "Treatment-specific β",
        "nmr_no_covariate": "Add at least 3 studies with a numeric value in Study data → Study characteristics (year, mean age, female %, follow-up), or use baseline risk (arm-level data only).",
        "nmr_coefficients": "Covariate coefficient(s) (change in {scale} per unit of the covariate)",
        "nmr_tau": "τ² without covariate {before:.4f} → with covariate {after:.4f}{explained}",
        "nmr_effects_at": "Effects vs {ref} at covariate = ",
        "nmr_curve": "Effect vs {ref} across the covariate range",
        "nmr_curve_treatment": "Treatment",
        "nmr_excluded": "Studies without a covariate value (left out): {ids}",
        "nmr_caveat": "Study-level covariates describe trial averages (ecological bias); baseline-risk regression is affected by regression to the mean. Treat as exploratory.",
        "funnel_order": "Order used (older / control → newer): {order}. Change it by reordering the Treatments table.",
        "funnel_reverse": "Reverse the order",
        "funnel_x": "Centred {scale} (study − comparison summary), newer vs older",
        "funnel_egger": "Egger's test (comparison-adjusted): intercept {a:.3f} (SE {se:.3f}), p = {p}",
        "funnel_asym": "Evidence of funnel asymmetry (p < 0.10): small studies may favour newer treatments (or older, depending on the side) - consider small-study effects / publication bias.",
        "funnel_sym": "No clear funnel asymmetry (p ≥ 0.10).",
        "loo_title": "Leave-one-study-out sensitivity",
        "loo_treatment": "Effect to follow",
        "loo_hint": "Each row refits the network without that study. Orange: removing the study changes whether the CI excludes the null for at least one treatment.",
        "loo_summary": "{treatment}: estimate ranges {low} to {high} across leave-one-out fits (full model {full}).",
        "open_file": "Open",
        "open_hint": "NMA file (.decide / .json) or study data (.csv)",
        "import_hint": "Import study data from RevMan (.rm5 / .csv) or Covidence (.csv)",
        "import_button": "Import from file",
        "import_done": "Imported {n} studies.",
        "load_example": "Load example network",
        "save": "Save",
        "save_as": "Save as",
        "filename": "Filename",
        "model_settings": "Model settings",
        "outcome": "Outcome type",
        "outcome_binary": "Binary (events / total)",
        "outcome_continuous": "Continuous (mean, SD, n)",
        "outcome_rate": "Count (events / person-time)",
        "outcome_contrast": "Reported effects (estimate + SE or CI)",
        "measure": "Effect measure",
        "effects_model": "Effects model",
        "fixed": "Fixed",
        "random": "Random",
        "reference": "Reference treatment",
        "lower_better": "Lower values are better",
        "lower_better_help": "On for harmful events (e.g. mortality OR); off when higher is better (e.g. response). Drives rankings.",
        "zero_correction": "Zero-cell correction",
        "ci_level": "Confidence level",
        "mcmc": "MCMC (Bayesian)",
        "mcmc_soon": "Off: the frequentist engine is used. Turn on (or choose Engine = Bayesian in Model settings) to edit these settings.",
        "mcmc_use": "Use the Bayesian engine",
        "report_title_ui": "Report export",
        "report_needs_run": "Run the NMA first to export a Word / Excel report.",
        "report_build": "Prepare report (Word + Excel)",
        "report_building": "Building the report…",
        "report_stale": "The results changed since the report was prepared - prepare it again.",
        "report_word": "Download Word report (.docx)",
        "report_excel": "Download Excel tables (.xlsx)",
        "rep_report_title": "Network meta-analysis report",
        "rep_generated": "Generated by DECIDE NMA Editor on {date}",
        "rep_sec_methods": "Methods",
        "rep_sec_network": "Network geometry",
        "rep_sec_characteristics": "Study characteristics",
        "rep_sec_results": "Relative effects",
        "rep_sec_league": "League table",
        "rep_sec_ranking": "Treatment ranking",
        "rep_sec_heterogeneity": "Heterogeneity",
        "rep_sec_inconsistency": "Inconsistency",
        "rep_sec_cinema": "Confidence in the evidence (CINeMA)",
        "rep_sec_contribution": "Contribution matrix",
        "rep_sec_absolute": "Absolute effects",
        "rep_sec_fit": "Model fit (Bayesian)",
        "rep_sec_prisma": "PRISMA-NMA reporting checklist",
        "rep_tab_prisma": "Table 11. PRISMA-NMA checklist (Hutton 2015): items marked 'Reported' are covered by this report; the remaining items are completed in the manuscript.",
        "sof_tab": "Summary of findings",
        "sof_intro": "GRADE-style Summary of Findings table for the network (Yepes-Nuñez 2019): direct evidence, relative and absolute effects, certainty from CINeMA and ranking. The layout chosen here is the one printed in the Word / Excel report.",
        "sof_layout": "Rows",
        "sof_layout_reference": "Each treatment vs {ref}",
        "sof_layout_all": "All pairs",
        "sof_layout_comparator": "Each treatment vs a chosen comparator",
        "sof_comparator": "Comparator",
        "rep_sec_sof": "Summary of findings",
        "rep_tab_sof": "Table 3b. Summary of findings (GRADE-style): direct evidence, relative and absolute effects, certainty of the evidence (CINeMA) and ranking.",
        "prisma_tab": "PRISMA-NMA",
        "prisma_intro": "Reporting checklist of the PRISMA extension for network meta-analyses (Hutton et al. 2015): 27 PRISMA items + 5 NMA items (S1–S5). The editor suggests a status for the items this report covers; edit Status and note where each item is reported in your manuscript. The checklist is included in the Word / Excel report.",
        "prisma_summary": "Reported {reported} · Partly {partly} · To do {todo} · Not applicable {na} of {total} items",
        "prisma_reset": "Reset to suggestions",
        "prisma_download": "Download checklist (CSV)",
        "rep_sec_references": "References",
        "rep_fig_network": "Figure 1. Network of included studies (node size = patients, edge width = studies).",
        "rep_fig_forest": "Figure 2. Effects of each treatment versus {ref} ({measure}, {pct} intervals).",
        "rep_tab_edges": "Table 1. Direct comparisons in the network.",
        "rep_tab_characteristics": "Table 2. Study characteristics by comparison (transitivity).",
        "rep_tab_effects": "Table 3. Network estimates versus {ref}.",
        "rep_tab_league": "Table 4. League table ({measure}): column treatment versus row treatment.",
        "rep_tab_ranking": "Table 5. Ranking.",
        "rep_tab_nodesplit": "Table 6. Node-splitting: direct versus indirect evidence.",
        "rep_tab_cinema": "Table 7. CINeMA confidence rating per comparison.",
        "rep_tab_contribution": "Table 8. Contribution (%) of each direct comparison to each network estimate.",
        "rep_tab_absolute": "Table 9. Absolute effects per 1000 patients (baseline risk {p0} with {ref}).",
        "rep_tab_fit": "Table 10. Model fit and convergence.",
        "rep_methods_freq": "A contrast-based network meta-analysis was fitted by generalised least squares ({effects} effects; between-study variance τ² by the generalised DerSimonian–Laird method of moments; multi-arm trials handled through their within-study covariance). Effects are reported as {measure} with {pct} confidence intervals; random-effects prediction intervals use a t distribution with {df} degrees of freedom. Treatments were ranked by P-score.",
        "rep_methods_bayes_normal": "A Bayesian network meta-analysis was fitted with a normal likelihood on the contrast scale ({effects} effects), vague priors N(0, {prior_sd}²) on the treatment effects and a {tau_prior} prior on the between-study standard deviation τ, sampling the posterior exactly (no MCMC convergence issues). Effects are posterior medians with {pct} credible intervals; treatments were ranked by SUCRA.",
        "rep_methods_bayes_exact": "A Bayesian network meta-analysis with the exact arm-level likelihood ({likelihood}) was fitted by Markov chain Monte Carlo following NICE DSU TSD 2 ({effects} effects; {chains} chains × {draws} draws after {warmup} warm-up iterations; vague priors N(0, {prior_sd}²) on the treatment effects and a {tau_prior} prior on τ). Convergence was checked with split R-hat and effective sample sizes; model fit with the residual deviance, pD and DIC. Effects are posterior medians with {pct} credible intervals; treatments were ranked by SUCRA.",
        "rep_methods_inconsistency": "Inconsistency was assessed globally with the design-by-treatment interaction test and locally by node-splitting ({split_method}).",
        "rep_methods_cinema": "Confidence in each network estimate was rated with the CINeMA framework (within-study bias, reporting bias, indirectness, imprecision, heterogeneity, incoherence), with the contribution matrix weighting study-level judgements.",
        "rep_methods_software": "Analyses were performed in DECIDE NMA Editor; the frequentist engine has been cross-checked against R netmeta.",
        "rep_het_text": "τ² = {tau2}, Q = {q} on {df} df (p = {p}), I² = {i2}.",
        "rep_dbt_text": "Design-by-treatment interaction test: Q = {q} on {df} df, p = {p}.",
        "rep_ume_text": "Consistency model DIC {dic_c} versus unrelated mean effects model DIC {dic_u} ({verdict}).",
        "rep_ume_ok": "no evidence of inconsistency",
        "rep_ume_bad": "the UME model fits better: evidence of inconsistency",
        "rep_no_data": "Not run.",
        "rep_study": "Study",
        "rep_treatment": "Treatment",
        "rep_comparison": "Comparison",
        "rep_studies": "Studies",
        "rep_patients": "Patients",
        "rep_estimate": "Estimate",
        "rep_ci": "CI",
        "rep_cri": "CrI",
        "rep_pi": "Prediction interval",
        "rep_score": "P-score",
        "rep_sucra": "SUCRA",
        "rep_rank": "Mean rank",
        "rep_direct": "Direct",
        "rep_indirect": "Indirect",
        "rep_difference": "Difference",
        "rep_confidence": "Confidence",
        "rep_settings": "Settings",
        "rep_sheet_data": "Study data",
        "split_method": "Node-splitting method",
        "split_side": "SIDE (Dias 2010, NICE TSD 4)",
        "split_sidde": "SIDDE (netmeta)",
        "split_method_help": "SIDE: the direct parameter is attached to the comparison inside every study that contains it, so a multi-arm study's other arms still feed the indirect estimate. SIDDE: whole studies (designs) containing the comparison form the direct evidence and the indirect estimate is back-calculated from the network estimate - the method of R netmeta (netsplit). They agree exactly when no multi-arm study contains the comparison.",
        "chip_random": "random effects",
        "chip_fixed": "fixed effect",
        "chip_reference": "reference",
        "chip_dres": "Residual deviance {dres:.1f} / {n} data points",
        "ee_empirical": "The Bayesian engine is on: {n} joint posterior draws per effect will be added as well, and the EE model's PSA will sample them directly (exact posterior shape and correlations) instead of the Lognormal approximation.",
        "colh_parameter": "Parameter",
        "colh_ci_lower": "CI lower",
        "colh_ci_upper": "CI upper",
        "colh_cri_lower": "CrI lower",
        "colh_cri_upper": "CrI upper",
        "colh_model": "Model",
        "colh_vs": "vs",
        "colh_estimate_scale": "Estimate (analysis scale)",
        "colh_pi_lower": "PI lower",
        "colh_pi_upper": "PI upper",
        "colh_variance": "Variance",
        "colh_study_removed": "Study removed",
        "colh_conclusion": "Conclusion changes for",
        "colh_design": "Design",
        "colh_study_ids": "Study IDs",
        "colh_year": "Year",
        "colh_mean_age": "Mean age",
        "colh_female": "Female %",
        "colh_follow_up": "Follow-up (months)",
        "colh_ratio_id": "Ratio (indirect / direct)",
        "colh_beta_for": "β for",
        "colh_network": "Network",
        "colh_all_studies": "All studies",
        "colh_without_rob": "Without high RoB",
        "colh_base": "Base",
        "colh_ume": "UME",
        "colh_nma": "NMA",
        "warn_zero_cell": "{id}: zero cell - {x} added to every cell of this study",
        "warn_excluded_zero": "{id}: excluded - no events (or all events) in every arm, so it carries no information on the ratio",
        "warn_zero_variance": "{id}: excluded - an arm has zero variance (0% or 100% risk with no zero-cell correction)",
        "warn_no_df": "No degrees of freedom for heterogeneity (as many contrasts as treatment effects): τ² set to 0, random effects = fixed effect",
        "warn_funnel_points": "Only {n} points: the funnel plot and Egger's test have little power below about 10.",
        "warn_no_residual_df": "No residual degrees of freedom: the regression fits the data exactly.",
        "warn_no_rob": "No risk-of-bias judgements entered (Study characteristics): studies without one count as 'Some concerns'.",
        "warn_low_acceptance": "Low MCMC acceptance in {which} - check R-hat / ESS and consider more draws.",
        "paste_label": "📋 Paste table",
        "paste_hint": "Copy the cells in Excel / Google Sheets (with or without the header row), paste below, then Replace or Append. You can also paste a block straight into a cell: it fills to the right and down.",
        "paste_replace": "Replace table",
        "paste_append": "Append rows",
        "paste_close": "Close",
        "paste_empty": "Nothing to paste.",
        "autosave_status": "Autosaved {time} (autosave/nma-autosave.json)",
        "autosave_found": "Unsaved work from the last session was found",
        "autosave_restore": "Restore",
        "autosave_discard": "Discard",
        "mcmc_running": "MCMC: {effects} effects…",
        "fit_fe_re": "Fixed vs random effects: the {better} model fits better by DIC (difference {gap:.1f}; below 3 the simpler fixed-effect model is usually preferred unless heterogeneity is expected).",
        "bayes_incons_title": "Bayesian inconsistency checks (TSD 4)",
        "bayes_incons_needs_exact": "Available with the Bayesian engine and the exact likelihood - run the NMA with Likelihood = Exact.",
        "bayes_split_title": "Bayesian node-splitting",
        "bayes_split_help": "One MCMC run per comparison that has both direct and indirect evidence: the direct studies get their own parameter, the rest of the network gives the indirect estimate. p = 2 × min(P(direct > indirect), P(direct < indirect)). Multi-arm studies share their baseline, so a conflict in one contrast can pull the direct estimate of another.",
        "bayes_split_run": "Run Bayesian node-splitting ({n} comparisons)",
        "bayes_split_hint": "Orange: p < 0.05. DIC of each split model vs {dic} for the consistency model - a clearly lower DIC also points to inconsistency.",
        "incons_col_direct_n": "Direct studies",
        "incons_col_indirect": "Indirect",
        "incons_col_ratio": "Ratio direct / indirect",
        "incons_col_difference": "Difference direct − indirect",
        "likelihood": "Likelihood",
        "likelihood_help": "Exact: binomial (OR logit / RR log / RD identity link), Poisson (rate ratio) or normal (MD; SMD on standardised means) likelihood on the arm-level data, by MCMC - the NICE DSU TSD 2 model; handles zero cells without continuity correction. Normal approximation: contrast-level normal likelihood, exact sampling in a second or two.",
        "lik_exact": "Exact (arm-level, MCMC)",
        "lik_normal": "Normal approximation (fast)",
        "mcmc_note_exact": "MCMC with the exact arm-level likelihood (TSD 2): per-study Laplace-proposal Metropolis-Hastings, Gibbs updates for the effects, slice sampling for τ. Chains run in parallel; the first 'Warm-up' iterations are discarded. Check R-hat ≤ 1.01 and ESS in Dashboard. Contrast-level studies enter with a normal likelihood. Takes a few seconds to a minute.",
        "lik_chip_exact": "exact {kind} likelihood",
        "lik_chip_normal": "normal approximation",
        "fit_title": "Model fit (TSD 2): residual deviance and leverage",
        "fit_hint": "Posterior mean residual deviance {dres:.1f} for {n} data points (a well-fitting model has Dres close to the number of data points), pD {pd:.1f}, DIC {dic:.1f}.",
        "fit_acceptance": "MCMC acceptance rates",
        "fit_x": "sign(r) × √(residual deviance)",
        "fit_y": "Leverage",
        "conv_title": "Convergence plots",
        "conv_help": "Trace: the sampled values of each chain over the kept iterations - chains should overlap and look like stationary noise. Density: the posterior of each chain - should coincide. Running mean: should settle quickly. The parameter with the largest R-hat is preselected.",
        "conv_parameter": "Parameter",
        "conv_chain": "chain",
        "conv_iteration": "Iteration (after warm-up)",
        "conv_trace": "Trace",
        "conv_density": "Posterior density per chain",
        "conv_running": "Running mean",
        "conv_summary": "Posterior median {median:.4g} ({lower:.4g} to {upper:.4g}) · R-hat {rhat:.3f} · ESS {ess:.0f}",
        "conv_warning": "Check {params}: R-hat > 1.01 or ESS < 400. Increase draws / warm-up or check the data.",
        "rep_fig_trace": "Figure 3. MCMC trace plots (one line per chain, warm-up discarded) with the marginal posterior density of each parameter.",
        "fit_leverage_note": "One point per arm (or per contrast-level study). Points outside the outer curve (x² + leverage = 3) fit poorly and/or influence the fit strongly - check their data.",
        "fit_outliers": "Poorly fitting or influential data points: {points}.",
        "ume_title": "Inconsistency: consistency vs unrelated mean effects (UME) model (TSD 4)",
        "ume_help": "The UME model gives every observed comparison its own effect, without assuming consistency. If it fits clearly better (lower DIC, points below the line), the direct and indirect evidence disagree.",
        "ume_run": "Run the UME model",
        "ume_consistency": "Consistency (NMA)",
        "ume_model": "Model",
        "ume_points": "Data points",
        "ume_worse": "DIC is {diff:.1f} lower with the UME model: evidence of inconsistency - look at the points below the line and the comparisons in the table.",
        "ume_ok": "The consistency model fits as well as the UME model (DIC difference {diff:.1f} ≤ 3): no evidence of inconsistency.",
        "ume_x": "Residual deviance, consistency model",
        "ume_y": "Residual deviance, UME model",
        "ume_devdev_note": "Each point is a data point; points well below the dashed line fit better without consistency - they point to where the inconsistency is.",
        "conv_title": "Conversion tools (median/IQR, CI, p-value…)",
        "conv_tool": "What do you need?",
        "conv_median": "Mean and SD from the median (with IQR and/or range)",
        "conv_se": "SE of an effect from its CI or p-value",
        "conv_sd": "SD of one arm from its SE or CI",
        "conv_combine": "Combine two arms of the same treatment",
        "conv_n": "n",
        "conv_median_v": "Median",
        "conv_mean": "Mean",
        "conv_ratio": "Ratio measure (OR, RR, HR…) - work on the log scale",
        "conv_from": "From",
        "conv_ci": "CI",
        "conv_p": "p-value",
        "conv_estimate": "Estimate",
        "conv_lower": "Lower",
        "conv_upper": "Upper",
        "conv_arm": "Arm",
        "conv_note_median": "Assumes a roughly symmetric distribution; with skewed data the estimate is less reliable. Record the conversion in the Source column.",
        "conv_copy": "Enter estimate + SE in the Contrast-level table (log scale for ratios).",
        "conv_fill_title": "Fill into table",
        "conv_fill_target": "Target row",
        "conv_fill_new": "+ New row (fill Study / Treatment afterwards)",
        "conv_fill_overwrite": "Overwrite the existing values",
        "conv_fill_button": "Fill into table",
        "conv_fill_existing": "This row already has {values}. Tick 'Overwrite' to replace them.",
        "conv_fill_done": "Filled {what} into row {row}.",
        "conv_combine_binary": "Binary outcomes: just add the events and the patients of the two arms.",
        "rank_tab": "Ranking across outcomes",
        "rank_intro": "P-score of each treatment for every analysis in this file (frequentist, each analysis's own effects model and 'lower is better' setting). Green = ranked high, red = low; – = not in that network.",
        "rank_single": "Add a second analysis (sidebar → Outcomes / analyses) to compare rankings across outcomes.",
        "rank_treatment": "Treatment",
        "rank_x": "X axis",
        "rank_y": "Y axis",
        "rank_scatter": "Treatments in the top-right rank well on both outcomes. P-scores ignore how large the differences are - read with the effect estimates.",
        "rank_failed": "Not included - {title}: {error}",
        "prior_title": "Prior sensitivity for τ (Bayesian, random effects)",
        "prior_help": "Refits the random-effects model under four priors for the between-study SD τ. Results that change a lot with the prior are driven by it rather than by the data - report this.",
        "prior_needs": "Choose the Bayesian engine to run this.",
        "prior_run": "Run prior sensitivity",
        "prior_col": "Prior for τ",
        "prior_current": "(current)",
        "abs_title": "Absolute effects and NNT",
        "abs_baseline": "Risk with {ref} (%)",
        "abs_hint": "Risk of the outcome with the reference over the study time frame - use the population of your decision problem where known.",
        "abs_observed": "Pooled risk in the reference arms of these trials: {pct}.",
        "abs_hr_note": "For HR / rate ratio: risk at the time point of interest, assuming proportional hazards.",
        "abs_col_treatment": "Comparison",
        "abs_col_ref": "With reference (per 1000)",
        "abs_col_risk": "With treatment (per 1000, CI)",
        "abs_col_diff": "Difference (per 1000, CI)",
        "abs_nnt_note": "NNTB = number needed to treat for one more patient to benefit; NNTH = to harm (rounded up). '∞' when the CI includes no effect (Altman 1998).",
        "bayes_not_run": "Bayesian engine selected but not run yet: click Run NMA to get SUCRA and rank probabilities. Showing frequentist results (P-score) for now.",
        "sucra_col_treatment": "Treatment",
        "sucra_col_mean_rank": "Mean rank",
        "sucra_col_best": "P(best)",
        "mcmc_note": "Exact posterior sampling (tau on a fine grid, effects drawn exactly given tau - the bayesmeta 'DIRECT' method): no burn-in or autocorrelation, chains are independent streams for the R-hat / ESS check. Within-study likelihood is normal on the contrast scale (same as the frequentist engine), not the exact binomial of NICE TSD 2.",
        "engine": "Engine",
        "frequentist": "Frequentist",
        "bayesian": "Bayesian",
        "tau_prior": "Prior for τ",
        "tau_informative": "Informative (Turner 2015 / Rhodes 2015)",
        "tau_prior_help": "Uniform / half-normal: vague. Informative: empirical predictive distribution for τ² (log-normal, Turner 2015 for binary outcomes on the log OR scale; log-t, Rhodes 2015 for continuous outcomes on the SMD scale) chosen by outcome type and intervention-comparison type - recommended for small networks, where a vague prior gives very wide intervals.",
        "tau_outcome_type": "Outcome type",
        "tau_comparison_type": "Comparison type",
        "tau_general_used": "general distribution",
        "tau_cell_missing": "This outcome × comparison cell has not been transcribed yet - the general distribution is used. Cells are read from priors/tau_priors.json.",
        "class_model": "Treatment classes",
        "class_none": "No class effects",
        "class_common": "Common class effect",
        "class_exchangeable": "Exchangeable class effects",
        "class_model_help": "Uses the Class column of the Treatments table (classes with two or more treatments in the network; exact-likelihood Bayesian engine only). Common: all treatments of a class share one effect. Exchangeable: each treatment's effect is drawn from a class distribution N(class mean, σ_class²), so treatments with little data are shrunk towards their class (NICE DSU TSD 3).",
        "class_sd_scale": "σ_class prior: half-normal scale",
        "class_unavailable": "Class effects need the exact-likelihood Bayesian engine and a class with two or more treatments (Treatments table, Class column).",
        "class_title": "Class effects",
        "class_common_help": "Treatments of a class share one effect versus the reference; SUCRA ranks the classes.",
        "class_exchangeable_help": "Class mean effect versus the reference; individual treatments are shrunk towards it. SUCRA ranks the classes.",
        "class_sd_text": "Between-treatment SD within a class (σ_class): median {median:.3f} (CrI {lower:.3f} to {upper:.3f}). Small values mean the treatments of a class behave alike.",
        "class_compare_button": "Fit the model without classes to compare DIC",
        "class_compare_text": "DIC with class effects {dic_c:.1f} vs without {dic_n:.1f} (difference {diff:+.1f}); pD {pd_c:.1f} vs {pd_n:.1f}. A lower DIC is better; differences below 3 are not meaningful.",
        "class_not_used": "Class effects were requested but not used:",
        "rep_sec_class": "Class effects",
        "rep_tab_class": "Table 10b. Class effects versus {ref} ({measure}, {pct} credible intervals) and SUCRA across classes.",
        "thr_tab": "Threshold analysis",
        "thr_intro": "How far could each piece of evidence move before the best treatment changes? Each observed contrast is shifted alone (all other data fixed); the invariant interval is where the recommendation stays the same (Phillippo 2018). Based on the frequentist GLS model (tau² held fixed); points whose limit lies inside the 95% interval of the data itself are fragile.",
        "thr_basis": "Decision basis",
        "thr_basis_effect": "Treatment effect",
        "thr_basis_nmb": "Net monetary benefit (CEA/CUA)",
        "thr_nmb_help": "Enter each treatment's cost and effect at the current NMA estimate, and how its effect and cost change per unit of its NMA effect (local sensitivities from your economic model, e.g. by re-running it at a slightly changed effect). The ICER table is informational: at every threshold the ICER of the new best treatment against the old one equals the willingness to pay.",
        "thr_wtp": "Willingness to pay per unit of effect",
        "thr_col_treatment": "Treatment",
        "thr_col_cost": "Cost",
        "thr_col_effect": "Effect (absolute)",
        "thr_col_d_effect": "dEffect / dd",
        "thr_col_d_cost": "dCost / dd",
        "thr_col_d_effect_help": "Change in this treatment's absolute effect per unit of its NMA effect on the analysis scale",
        "thr_col_d_cost_help": "Change in this treatment's cost per unit of its NMA effect on the analysis scale",
        "thr_best": "Current best treatment: **{best}** (order: {order}).",
        "thr_icer_title": "Cost-effectiveness against the reference",
        "thr_icer_help": "ICER = incremental cost / incremental effect versus the reference treatment.",
        "thr_inc_cost": "Incremental cost",
        "thr_inc_effect": "Incremental effect",
        "thr_scope": "Shift",
        "thr_scope_studies": "Each study contrast",
        "thr_scope_network": "Each network estimate",
        "thr_study": "Study",
        "thr_comparison": "Comparison",
        "thr_estimate": "Estimate",
        "thr_lower": "Lower limit",
        "thr_lower_best": "Best treatment below",
        "thr_upper": "Upper limit",
        "thr_upper_best": "Best treatment above",
        "thr_fragile": "Within 95% interval of the data",
        "thr_never": "never",
        "thr_download": "Download threshold table (CSV)",
        "thr_fragile_yes": "Fragile",
        "thr_fragile_no": "Robust",
        "thr_axis": "Estimate (natural scale)",
        "thr_chart_help": "Dot = observed estimate; line = invariant interval (the recommendation is unchanged inside it). Red = a limit lies within the 95% interval of the estimate.",
        "yes": "yes",
        "no": "no",
        "rep_sec_threshold": "Threshold analysis",
        "rep_tab_threshold": "Table 10c. Threshold analysis ({basis}): for each study contrast, the value at which the best treatment (currently {best}) changes, with the treatment that takes over.",
        "rep_methods_threshold": "Threshold analysis (Phillippo et al. 2018) was run on the {basis} basis: each observed contrast was shifted alone, all other data fixed, and the limits of its invariant interval, beyond which the best treatment changes, were computed from the network estimates, which are linear in the data.",
        "rep_basis_effect": "treatment effect",
        "rep_basis_nmb": "net monetary benefit",
        "cmp_tab": "Component NMA",
        "cmp_intro": "Additive component NMA (Rücker 2020): a treatment is a set of components and its effect is the sum of the component effects. Enter each treatment's components, separated by +; leave a treatment empty to treat it as a single component (the reference is the empty set).",
        "cmp_col_components": "Components",
        "cmp_col_help": "e.g. CBT + drug",
        "cmp_component": "Component",
        "cmp_none": "Enter components for at least one treatment to fit the additive model.",
        "cmp_effects_title": "Effect of each component",
        "cmp_treatments_title": "Treatments: additive model vs standard NMA",
        "cmp_additive": "Additive model",
        "cmp_standard": "Standard NMA",
        "cmp_additivity": "Test of additivity: Q = {q} on {df} df, p = {p} (a small p suggests components interact). Frequentist only.",
        "cmp_settings_hint": "Estimation method (frequentist or Bayesian) and interaction terms: NMA settings > Component NMA.",
        "cmp_interactions_used": "Interaction terms in the model: {items}.",
        "cmp_model": "Estimation",
        "cmp_model_freq": "Frequentist",
        "cmp_model_bayes": "Bayesian (MCMC settings from the sidebar)",
        "cmp_interactions": "Interaction terms (component pairs whose combined effect is not the sum)",
        "cmp_bayes_title": "Implied treatment effects and SUCRA (posterior median, credible interval)",
        "cmp_bayes_diag": "Convergence (R-hat, ESS)",
        "rep_sec_components": "Component network meta-analysis",
        "rep_tab_components": "Table 10d. Additive component NMA: effect of each component ({measure}) and treatment effects versus {ref}; test of additivity Q = {q} on {df} df (p = {p}).",
        "rep_methods_components": "An additive component network meta-analysis (Rücker et al. 2020) was fitted: the effect of a treatment versus the reference is the sum of the effects of its components. Additivity was assessed by comparing the additive model with the standard network meta-analysis.",
        "des_tab": "Non-randomised evidence",
        "des_intro": "Tick the studies that are not randomised. The network is re-estimated with them excluded, at full weight, and down-weighted by alpha (their variances are divided by alpha), so you can see how much the conclusions depend on them. This is a sensitivity analysis; your main analysis is unchanged. Frequentist model.",
        "des_col_nrs": "Non-randomised",
        "des_none": "Tick at least one non-randomised study.",
        "des_alphas": "Down-weighting factors (alpha)",
        "des_note": "Cells show the effect versus the reference (confidence interval). alpha = 0.25 means a non-randomised study counts as a quarter of its size.",
        "rep_sec_design": "Non-randomised evidence",
        "rep_tab_design": "Table 10e. Sensitivity to non-randomised studies ({nrs}): effects versus {ref} ({measure}, {pct} confidence intervals) with those studies excluded, at full weight, and down-weighted.",
        "rep_methods_design": "Studies marked as non-randomised were handled in a sensitivity analysis: excluded, included at full weight, and down-weighted by dividing their variances by a weight alpha.",
        "tv_tab": "Time-varying HR",
        "tv_intro": "Hazard ratios by time interval (e.g. 0-6, 6-12, 12+ months). Enter one row per study, interval and contrast, with log HR and SE, or HR and its 95% CI. A separate network meta-analysis is run for each interval, treating the intervals of one study as independent (an approximation: they share patients). A multi-arm study reporting two contrasts in one interval also needs the baseline SE.",
        "tv_from": "From",
        "tv_to": "To",
        "tv_lower": "HR lower 95%",
        "tv_upper": "HR upper 95%",
        "tv_baseline_se": "Baseline SE",
        "tv_baseline_help": "SE of the log hazard in the shared comparator arm; only for multi-arm studies",
        "tv_none": "Enter at least one complete row (study, treatment, versus, from, to, and HR data).",
        "tv_reference": "Reference treatment",
        "tv_unit": "Time unit",
        "tv_results": "Hazard ratio versus the reference, by interval",
        "tv_chart_help": "Each bar spans a time interval at the network hazard ratio (log axis).",
        "tv_ph_title": "Proportional-hazards check",
        "tv_ph_help": "Cochran's Q of the interval log HRs (independence assumed). A small p-value suggests the HR changes over time; a large one is not proof of proportional hazards (low power).",
        "tv_level": "Level",
        "tv_item": "Contrast / treatment",
        "tv_intervals": "Intervals",
        "tv_flag": "HR changes over time?",
        "tv_possibly": "possibly",
        "tv_no_evidence": "no evidence",
        "rep_sec_piecewise": "Time-varying hazard ratios",
        "rep_tab_piecewise": "Table 10f. Piecewise hazard ratios versus {ref} ({pct} confidence intervals) by time interval, from a separate network meta-analysis per interval; intervals of one study are treated as independent.",
        "rep_methods_piecewise": "Hazard ratios were analysed by time interval: a separate contrast-based network meta-analysis was fitted to the study hazard ratios of each interval, treating the intervals of a study as independent. Proportional hazards were assessed by Cochran's Q of the interval log hazard ratios.",
        "sv_tab": "Survival NMA",
        "sv_intro": "Network meta-analysis of survival curves: exponential, Weibull, Gompertz or fractional-polynomial hazards fitted to interval event counts and person-time (Poisson likelihood, Bayesian, a treatment effect on every hazard coefficient). Counts can be typed in or reconstructed from published Kaplan-Meier curves (Guyot 2012). Compare models with the DIC, check the curves against the follow-up, and export the curves to the economic model.",
        "sv_data_title": "Interval data",
        "sv_data_help": "One row per study, arm and time interval: events and person-time in the interval (same time unit as the interval limits).",
        "sv_km_title": "Reconstruct interval data from Kaplan-Meier curves",
        "sv_km_help": "Digitise each published curve (time, survival) and copy the numbers at risk printed under it (the first row is time 0 with the arm size). The counts are rebuilt with the Guyot algorithm and written to the interval table above. Accuracy depends on the digitised points; use real IPD when available.",
        "sv_km_points": "Curve points",
        "sv_km_risk": "Numbers at risk",
        "sv_breaks": "Interval limits (start at 0)",
        "sv_breaks_error": "Interval limits must be increasing numbers starting at 0",
        "sv_reconstruct": "Reconstruct and fill the interval table",
        "sv_none": "Enter interval data for at least two arms (or reconstruct it from curves).",
        "sv_time_factor": "Model time unit in years",
        "sv_time_factor_help": "For the PSM export: 1/12 for months, 1 for years",
        "sv_model": "Model",
        "sv_warmup": "Warm-up",
        "sv_draws": "Draws per chain",
        "sv_run": "Run this model",
        "sv_run_all": "Run exponential, Weibull and Gompertz",
        "sv_running": "Sampling...",
        "sv_compare": "Model comparison (DIC, lower is better)",
        "sv_show": "Show model",
        "sv_effects_title": "Treatment effects on the hazard coefficients",
        "sv_effects_help": "Each row is the effect of a treatment versus the reference on one coefficient of the log-hazard. For Weibull the first is the log-hazard at the end of follow-up and the second the change of the log shape.",
        "sv_baseline": "Baseline hazard of the reference",
        "sv_baseline_pooled": "Pooled over studies with a reference arm",
        "sv_horizon": "Curve horizon (time units)",
        "sv_survival": "Survival",
        "sv_chart_help": "Median curves with credible bands; the dashed line marks the end of follow-up: beyond it the curves are extrapolations.",
        "sv_export_title": "Export to the economic model",
        "sv_export_help": "Survival table: median curves. PSA draws: one posterior sample per draw, aligned across treatments. PSM parameters: closed-form parameters for the PSM node (exponential, Weibull, Gompertz).",
        "sv_dl_table": "Survival curves (CSV)",
        "sv_psa_n": "Number of PSA draws",
        "sv_dl_psa": "PSA curve draws (CSV)",
        "sv_dl_psm": "PSM parameters (CSV)",
        "sv_psm_note": "Fractional-polynomial curves have no closed form in the PSM node: use the survival table or the PSA draws.",
        "rep_sec_survival": "Survival network meta-analysis",
        "rep_tab_survival": "Table 10g. Survival NMA: model comparison (deviance information criterion, lower is better) and treatment effects on the coefficients of the log-hazard of the selected model ({model}) versus {ref}.",
        "rep_methods_survival": "A survival network meta-analysis was fitted to interval event counts and person-time (Poisson likelihood; data reconstructed from published Kaplan-Meier curves where indicated) with exponential, Weibull, Gompertz or fractional-polynomial hazards (Jansen 2011), a treatment effect on every hazard coefficient, and Metropolis-adjusted Langevin sampling; models were compared with the deviance information criterion.",
        "pa_tab": "Population adjustment",
        "pa_intro": "Adjust for differences in patient populations when you have individual patient data (IPD) for some trials and only aggregate data for others: MAIC re-weights one IPD trial to match another trial's covariates; ML-NMR fits a regression on individual and aggregate data together (Phillippo 2020). Both assume that every effect modifier is measured; check this with clinical input.",
        "pa_privacy": "IPD stay in this session only: they are not saved into the file, the autosave or the report (the report holds summary tables).",
        "pa_upload": "IPD file (CSV, one row per patient)",
        "pa_need_ipd": "Upload an IPD file to begin.",
        "pa_loaded": "{rows} patients, {cols} columns loaded.",
        "pa_preview": "Preview the first rows",
        "pa_maic_tab": "MAIC",
        "pa_mlnmr_tab": "ML-NMR",
        "pa_arm_col": "Treatment arm column",
        "pa_outcome_type": "Outcome type",
        "pa_type_binary": "Binary",
        "pa_type_continuous": "Continuous",
        "pa_type_survival": "Time to event",
        "pa_two_arms": "The arm column needs at least two different values.",
        "pa_treatment": "Treatment",
        "pa_comparator": "Comparator",
        "pa_event_col": "Event column (0/1)",
        "pa_value_col": "Outcome column",
        "pa_time_col": "Time column",
        "pa_cov_cols": "Covariates to match (numeric)",
        "pa_pick_covs": "Choose at least one covariate.",
        "pa_target": "Target population (aggregate trial)",
        "pa_target_help": "Enter the covariate means reported for the aggregate trial (proportions for 0/1 covariates) and, optionally, the SDs.",
        "pa_match_sd": "Also match the standard deviations (where entered)",
        "pa_se_method": "Standard error",
        "pa_se_auto": "Robust (bootstrap for time to event)",
        "pa_se_boot": "Bootstrap (re-estimates the weights)",
        "pa_run": "Run",
        "pa_running": "Fitting...",
        "pa_ess": "Effective sample size",
        "pa_covariate": "Covariate",
        "pa_unweighted": "IPD, unweighted",
        "pa_weighted": "IPD, weighted",
        "pa_target_col": "Target",
        "pa_analysis": "Analysis",
        "pa_unadjusted": "Unadjusted",
        "pa_adjusted": "Adjusted (MAIC)",
        "pa_balance": "Covariate balance",
        "pa_effect": "Effect",
        "pa_scale_binary": "odds ratio",
        "pa_scale_continuous": "mean difference",
        "pa_scale_survival": "hazard ratio",
        "pa_weight": "Weight",
        "pa_anchor_title": "Anchored indirect comparison",
        "pa_anchor_help": "If the aggregate trial compares another treatment with the same comparator, enter its effect and 95% interval on the same scale to get the adjusted indirect comparison of the IPD treatment against it (Bucher). Without a common comparator an unanchored comparison is only credible if all prognostic factors and effect modifiers are matched, which is rarely the case.",
        "pa_agg_effect": "Aggregate trial effect",
        "pa_agg_lower": "Lower 95%",
        "pa_agg_upper": "Upper 95%",
        "pa_agg_positive": "Ratio measures need positive values.",
        "pa_anchor_result": "Adjusted indirect comparison: {est} (95% CI {lo} to {hi})",
        "pa_ml_intro": "Binary outcome. IPD arms enter with the individual likelihood; aggregate arms (events, n, covariate means and SDs) are integrated over their covariate distribution. Effects are reported at a covariate profile.",
        "pa_study_col": "Study column",
        "pa_em_cols": "Effect modifiers",
        "pa_agg_title": "Aggregate arms",
        "pa_agg_help": "One row per aggregate-trial arm: study, treatment, events, n and the covariate means (proportions for 0/1) and SDs.",
        "pa_shared": "Shared effect modifier across treatments",
        "pa_shared_help": "One interaction per covariate for all active treatments versus the reference (a common assumption when data are sparse); untick for one per treatment.",
        "pa_profile": "Covariate profile",
        "pa_parameter": "Parameter",
        "pa_or": "Odds ratio vs reference",
        "pa_ml_effects": "Conditional odds ratios at the profile",
        "pa_coefficients": "Coefficients (log-odds scale, covariates centred at the pooled IPD mean)",
        "pa_ml_note": "Maximum likelihood with study-specific intercepts and a normal approximation; a random-effects or Bayesian version is not included. With no IPD the interactions rest on between-study variation only.",
        "rep_sec_popadj": "Population adjustment",
        "rep_tab_maic": "Table 10h. Matching-adjusted indirect comparison: covariate balance before and after weighting and the adjusted effect (effective sample size shown); individual patient data are not included in this report.",
        "rep_tab_mlnmr": "Table 10i. Multilevel network meta-regression (binary outcome, maximum likelihood): coefficients and conditional odds ratios versus the reference at the stated covariate profile.",
        "rep_methods_maic": "A matching-adjusted indirect comparison (Signorini 2010; NICE DSU TSD 18) re-weighted the individual patient data by the method of moments so that the weighted covariate distribution matched the aggregate trial; the effect was estimated in the weighted sample with the stated standard-error method.",
        "rep_methods_mlnmr": "A multilevel network meta-regression (Phillippo 2020) combined individual and aggregate data: aggregate arms were modelled by integrating the individual-level model over their covariate distributions (quasi-Monte Carlo), and the effects were estimated by maximum likelihood.",
        "dr_tab": "Dose-response",
        "dr_intro": "Dose-response network meta-analysis (MBNMA): the effect depends on the dose of each agent through a function (linear, Emax, ...), estimated from all arms; a dose-0 arm is the placebo. One row per arm: study, agent, dose and either mean and SE (continuous) or events and n (binary, log odds scale). Maximum likelihood, study-specific intercepts, no random effects between studies.",
        "dr_none": "Enter at least four arms (with a dose-0 arm in some studies).",
        "dr_groups_title": "Class effects: share a curve between agents",
        "dr_groups_help": "Agents with the same group name share one set of dose-response parameters (a common class effect). Leave as is for one curve per agent.",
        "dr_mixed": "Use either mean and SE or events and n in all arms, not both.",
        "dr_functions": "Dose-response functions to fit",
        "dr_compare": "Comparison of functional forms",
        "dr_function": "Function",
        "dr_deviance": "Residual deviance",
        "dr_fit_p": "Fit p",
        "dr_dispersion": "Dispersion",
        "dr_param_help": "Emax: theta0 = maximum effect, theta1 = log ED50. Hill: theta2 = log of the Hill coefficient. Exponential: theta1 = log of the dose scale.",
        "dr_dose": "Dose",
        "dr_effect_axis_continuous": "Effect vs dose 0 (mean difference)",
        "dr_effect_axis_binary": "Effect vs dose 0 (log odds ratio)",
        "dr_chart_help": "Curves with 95% bands; dots are observed arm effects against the placebo arm of the same study; dashed lines mark the highest dose studied for each agent, beyond which the curve is an extrapolation.",
        "rep_sec_dose": "Dose-response network meta-analysis",
        "rep_tab_dose": "Table 10j. Dose-response MBNMA: comparison of functional forms (AIC, lower is better) and parameters of the selected function ({function}). Theta are the function's parameters; ED50 and Hill terms are on the log scale.",
        "rep_methods_dose": "A model-based network meta-analysis of dose-response (Mawdsley 2016) was fitted to arm-level data by maximum likelihood with study-specific intercepts, so that only within-study dose contrasts inform the curves; functional forms were compared with the Akaike information criterion.",
        "chip_tau_prior": "τ prior: {label}",
        "tau_uniform": "Uniform(0, U)",
        "tau_half_normal": "Half-normal(0, s)",
        "tau_upper": "U",
        "tau_scale": "s",
        "prior_sd": "Prior SD for effects",
        "sucra": "Rankings (SUCRA)",
        "rankogram": "Rank probabilities (1 = best)",
        "bayes_diag": "Posterior summary and convergence",
        "bayes_fit": "Model fit",
        "bayes_note_results": "Bayesian results: posterior medians with {pct} credible intervals. Node-splitting and leave-one-out use the frequentist engine.",
        "chains": "Chains",
        "seed": "Seed",
        "warmup": "Warm-up",
        "draws": "Draws / chain",
        "treatments": "Treatments",
        "add_row": "+ Add row",
        "delete_row": "Delete row",
        "undo": "↶ Undo (⌘Z)",
        "redo": "↷ Redo (⇧⌘Z)",
        "treatments_help": "Aliases: comma-separated spellings merged into this name. Colour: #hex, used when Node colour = Custom.",
        "study_data": "Study data",
        "arm_level": "Arm-level",
        "contrast_level": "Contrast-level (HR / OR / MD)",
        "contrast_help": "Give estimate + SE on the analysis scale (log for ratios), or effect + lower + upper on the natural scale. Multi-arm studies: same comparator in every row and baseline_se.",
        "network": "Network",
        "view": "View",
        "node_colour": "Node colour",
        "by_class": "By class",
        "single": "Single",
        "custom": "Custom",
        "size_by": "Size by",
        "patients": "Patients",
        "studies": "Studies",
        "equal": "Equal",
        "selected_edge": "Inspect comparison",
        "no_edge": "(none)",
        "direct": "Direct",
        "indirect_via": "Indirect paths via",
        "none": "none",
        "network_ok": "Network connected",
        "fix_errors": "Fix these before running",
        "add_missing": "Add missing treatments to the list",
        "warnings": "Notes",
        "check_title": "Network check",
        "results_title": "Base case results",
        "details_title": "Model details",
        "stale": "The data changed since this run - click Run NMA again.",
        "not_run": "Click Run NMA to fit the model.",
        "forest": "Forest plot vs reference",
        "rankings": "Rankings (P-score)",
        "league": "League table",
        "league_hint": "Cell = effect of the column treatment vs the row treatment; ordered by P-score.",
        "heterogeneity": "Heterogeneity",
        "observations": "Study contrasts entered in the model",
        "excluded": "Excluded studies",
        "download_csv": "Download CSV",
        "illustrative": "Example data are illustrative, not real trial results.",
        "parse_error": "Could not read the data",
        "cinema_tab": "Confidence (CINeMA)",
        "contrib_tab": "Contribution matrix",
        "cinema_intro": "Confidence in each network estimate (CINeMA / GRADE-NMA): six domains rated automatically from the results already computed. Hover a chip to see the reason; * = changed by hand.",
        "cinema_margin_ratio": "Clinically important effect ({measure})",
        "cinema_margin_ratio_help": "Range of equivalence 1/m to m, e.g. 1.25 → 0.80–1.25. Used for imprecision and heterogeneity.",
        "cinema_margin_diff": "Clinically important difference ({measure})",
        "cinema_margin_diff_help": "Range of equivalence −m to m. 0 = no margin: judge only against the line of no effect.",
        "cinema_rule": "Risk of bias / indirectness rule",
        "cinema_rule_average": "Average",
        "cinema_rule_majority": "Majority",
        "cinema_rule_help": "Average: study ratings (Low 1, Some 2, High 3) weighted by each study's contribution to the estimate (< 1.5 no concerns, < 2.5 some, else major). Majority: the rating that contributes most.",
        "cinema_needs_run": "Click Run NMA first - CINeMA uses the NMA, inconsistency and funnel-plot results.",
        "cinema_freq_note": "CINeMA and the contribution matrix use the frequentist fit.",
        "cinema_edit": "Review or change a judgement",
        "cinema_comparison": "Comparison",
        "cinema_auto": "Auto: {level}",
        "cinema_note": "Reason for the change (kept in the report)",
        "cinema_reset": "Reset this comparison to automatic",
        "cinema_rules": "How the automatic rating works",
        "cinema_rules_text": "- **Within-study bias** - risk of bias of each study (Study characteristics), weighted by its contribution to the estimate (contribution matrix). Studies without a rating count as *Some concerns*.\n- **Reporting bias** - Egger's test on the comparison-adjusted funnel plot: p < 0.10 → *Suspected*.\n- **Indirectness** - study-level indirectness (Study characteristics), weighted like risk of bias; studies without a rating count as *Low*.\n- **Imprecision** - CI against the range of equivalence: crosses one boundary → *Some*, both → *Major*.\n- **Heterogeneity** - prediction interval vs CI: the PI crosses one more boundary → *Some*, two more → *Major*.\n- **Incoherence** - node-split p (direct + indirect evidence), otherwise the design-by-treatment test: p ≤ 0.10 → *Some*, p ≤ 0.01 → *Major*.\n\nConfidence starts at **High** and drops one level per *Some concerns* / *Suspected* and two per *Major concerns*. The domains overlap (imprecision, heterogeneity and incoherence), so review the suggestion and change it with a reason where needed (Nikolakopoulou et al. 2020, PLoS Med).",
        "cinema_download": "Download CINeMA report (CSV)",
        "col_comparison": "Comparison",
        "col_estimate": "Estimate (CI)",
        "col_pi": "Prediction interval",
        "col_direct": "Direct %",
        "col_confidence": "Confidence",
        "dom_within_study_bias": "Within-study bias",
        "dom_reporting_bias": "Reporting bias",
        "dom_indirectness": "Indirectness",
        "dom_imprecision": "Imprecision",
        "dom_heterogeneity": "Heterogeneity",
        "dom_incoherence": "Incoherence",
        "lvl_no": "No concerns",
        "lvl_some": "Some concerns",
        "lvl_major": "Major concerns",
        "lvl_undetected": "Undetected",
        "lvl_suspected": "Suspected",
        "lvl_high": "High",
        "lvl_moderate": "Moderate",
        "lvl_low": "Low",
        "lvl_very_low": "Very low",
        "contrib_intro": "Share (%) of each direct comparison (columns) in each network estimate (rows), from the hat matrix split into evidence-flow paths (Papakonstantinou et al. 2018). Orange outline = the comparison's own direct evidence.",
        "contrib_estimate": "Network estimate",
        "contrib_studies": "Study contributions to",
        "contrib_study": "Study",
        "contrib_share": "Contribution (%)",
        "contrib_rob": "Risk of bias",
        "contrib_download": "Download contribution matrix (CSV)",
        "outcomes": "Outcomes / analyses",
        "outcomes_help": "Several analyses of the same network in one file (e.g. mortality, hospitalisation, adverse events). Treatments and study characteristics are shared; study data, model settings and CINeMA judgements belong to each analysis.",
        "outcome_active": "Active analysis",
        "outcome_title": "Name of this analysis",
        "outcome_new": "New analysis",
        "outcome_new_name": "Name",
        "outcome_copy": "Copy study data from the current analysis",
        "outcome_add": "Add",
        "outcome_delete": "Delete this analysis",
        "outcome_default": "Analysis {n}",
    },
    "vi": {
        "title": "NMA Editor",
        "subtitle": "© 2026 Trang Nguyen",
        "auto_arrange": "Sắp xếp lại (đặt lại góc nhìn mạng lưới)",
        "check_network": "Kiểm tra model",
        "run_nma": "Chạy NMA",
        "inconsistency": "Inconsistency",
        "inconsistency_help": "Node-splitting (trực tiếp vs gián tiếp) và kiểm định design-by-treatment",
        "incons_title": "Inconsistency",
        "incons_global": "Kiểm định toàn mạng (design-by-treatment interaction)",
        "incons_row_total": "Tổng (mô hình consistency)",
        "incons_row_within": "Trong cùng thiết kế (heterogeneity)",
        "incons_row_between": "Giữa các thiết kế (inconsistency)",
        "incons_row_between_re": "Giữa các thiết kế, sau khi tính heterogeneity (τ² trong thiết kế)",
        "incons_flag": "Có dấu hiệu inconsistency (p < 0.05) - xem các so sánh được đánh dấu và đặc điểm nghiên cứu trước khi dùng ước lượng mạng lưới.",
        "incons_ok": "Không có inconsistency có ý nghĩa thống kê (p ≥ 0.05). Lực kiểm định thấp khi ít vòng kín - đọc cùng bảng node-split.",
        "incons_split": "Node-splitting: bằng chứng trực tiếp vs gián tiếp",
        "incons_split_hint": "Dòng màu cam: p < 0.05. Ratio = gián tiếp / trực tiếp (thước đo tỷ số) hoặc chênh lệch = trực tiếp - gián tiếp.",
        "incons_none": "Không có so sánh nào vừa có bằng chứng trực tiếp vừa có đường gián tiếp độc lập (không có vòng kín), nên không node-split được.",
        "incons_not_split": "Chỉ có trực tiếp hoặc chỉ có gián tiếp (không tách được)",
        "incons_designs": "Q theo thiết kế",
        "prediction_note": "Thanh nhạt: khoảng dự đoán {pct} (random effects, t với {df} bậc tự do).",
        "dashboard": "Dashboard",
        "send_to_ee": "Gửi sang mô hình EE",
        "send_to_ee_help": "Thêm hiệu quả tương đối từ NMA (kèm phân phối PSA và tương quan) thành tham số của mô hình Markov / PSM / Tree / Microsimulation",
        "ee_title": "Evidence Set → mô hình EE",
        "ee_target": "Mô hình đích",
        "ee_anchor": "So với (nhánh nền của mô hình EE)",
        "ee_treatments": "Treatment cần xuất",
        "ee_prefix": "Tiền tố tên (tuỳ chọn)",
        "ee_add_psa": "Thêm cả phân phối PSA + tương quan",
        "ee_preview": "Các tham số sẽ được thêm hoặc cập nhật",
        "ee_corr": "{n} tương quan giữa các hiệu quả này sẽ được thêm vào PSA của mô hình (chúng dùng chung bằng chứng của {anchor}).",
        "ee_apply": "Thêm vào {model}",
        "ee_done": "{model}: đã thêm {added}; đã cập nhật {updated}. Mở {model} → Parameters để dùng.",
        "ee_formula": "Cách áp dụng trong mô hình (biểu thức DECIDE; p_base = tham số xác suất nền của {anchor}):",
        "ee_download": "Tải evidence set (JSON)",
        "ee_no_target": "Mô hình này chưa được mở trong phiên làm việc - mở nó một lần rồi quay lại.",
        "more": "Phân tích thêm",
        "more_help": "Đánh giá độ tin cậy CINeMA, contribution matrix, network meta-regression và funnel plot hiệu chỉnh so sánh",
        "more_title": "Phân tích thêm",
        "nmr_tab": "Network meta-regression",
        "funnel_tab": "Funnel plot",
        "nmr_covariate": "Covariate",
        "nmr_baseline": "Nguy cơ nền (nhánh reference, thang phân tích)",
        "nmr_interaction": "Tương tác",
        "nmr_shared": "Chung (một β cho mọi treatment so với reference)",
        "nmr_specific": "β riêng từng treatment",
        "nmr_no_covariate": "Cần ít nhất 3 nghiên cứu có giá trị số trong Dữ liệu nghiên cứu → Đặc điểm nghiên cứu (năm, tuổi, % nữ, follow-up), hoặc dùng nguy cơ nền (chỉ với dữ liệu arm-level).",
        "nmr_coefficients": "Hệ số covariate (thay đổi {scale} trên mỗi đơn vị covariate)",
        "nmr_tau": "τ² không có covariate {before:.4f} → có covariate {after:.4f}{explained}",
        "nmr_effects_at": "Hiệu quả so với {ref} tại covariate = ",
        "nmr_curve": "Hiệu quả so với {ref} theo khoảng giá trị covariate",
        "nmr_curve_treatment": "Treatment",
        "nmr_excluded": "Nghiên cứu không có giá trị covariate (bị loại): {ids}",
        "nmr_caveat": "Covariate cấp nghiên cứu là giá trị trung bình của thử nghiệm (ecological bias); hồi quy theo nguy cơ nền bị ảnh hưởng bởi regression to the mean. Nên coi là phân tích thăm dò.",
        "funnel_order": "Thứ tự đang dùng (cũ / đối chứng → mới): {order}. Đổi bằng cách sắp xếp lại bảng Treatments.",
        "funnel_reverse": "Đảo ngược thứ tự",
        "funnel_x": "{scale} đã căn giữa (nghiên cứu − ước lượng của so sánh), mới so với cũ",
        "funnel_egger": "Kiểm định Egger (hiệu chỉnh so sánh): intercept {a:.3f} (SE {se:.3f}), p = {p}",
        "funnel_asym": "Có dấu hiệu funnel bất đối xứng (p < 0.10): nghiên cứu nhỏ có thể nghiêng về treatment mới (hoặc cũ, tuỳ phía) - cân nhắc small-study effects / publication bias.",
        "funnel_sym": "Không thấy funnel bất đối xứng rõ (p ≥ 0.10).",
        "characteristics": "Đặc điểm nghiên cứu",
        "characteristics_help": "Không bắt buộc, mỗi nghiên cứu một dòng: dùng cho bảng transitivity (Kiểm tra mạng) và phân tích độ nhạy theo nguy cơ sai lệch (Dashboard) và CINeMA (Phân tích thêm). Risk of bias và Indirectness: Low / Some / High.",
        "chars_vars_title": "Biến (cột) của bảng này",
        "chars_vars_help": "Study, Risk of bias, Indirectness và Notes cố định. Thêm, đổi tên, đổi kiểu hoặc xoá các biến còn lại; biến số được tóm tắt theo từng so sánh trong bảng transitivity và dùng làm covariate cho network meta-regression.",
        "chars_var_label": "Biến",
        "chars_var_type": "Kiểu",
        "chars_type_number": "Số",
        "chars_type_text": "Chữ",
        "chars_var_new": "Biến mới",
        "chars_var_add": "Thêm biến",
        "chars_var_delete": "Xoá '{name}' (mất giá trị đã nhập)",
        "chars_var_reset": "Về bộ biến mặc định",
        "transitivity": "Transitivity: đặc điểm nghiên cứu theo từng so sánh",
        "transitivity_hint": "Trung bình (min–max) của các nghiên cứu trong từng so sánh trực tiếp. Các so sánh khác nhau nhiều về yếu tố điều chỉnh hiệu quả (tuổi, mức độ bệnh, thời gian theo dõi...) làm so sánh gián tiếp kém tin cậy.",
        "transitivity_plot": "Biểu đồ transitivity: giá trị của từng nghiên cứu theo so sánh",
        "transitivity_plot_hint": "Mỗi chấm là một nghiên cứu, tô màu theo nguy cơ sai lệch; thanh là khoảng tứ phân vị, vạch đậm là trung vị. Nếu các chấm của những so sánh nằm ở chỗ khác nhau thì so sánh gián tiếp dựa trên chúng kém đáng tin hơn.",
        "transitivity_variable": "Biến",
        "transitivity_plot_empty": "Chưa có đặc điểm nghiên cứu dạng số (Study data → Characteristics).",
        "transitivity_text_counts": "Đặc điểm dạng chữ: số nghiên cứu theo nhóm và so sánh",
        "rep_fig_transitivity": "Hình 1b. Đặc điểm nghiên cứu theo so sánh (transitivity): mỗi chấm là một nghiên cứu, tô màu theo nguy cơ sai lệch; thanh là khoảng tứ phân vị, vạch đậm là trung vị.",
        "rob_title": "Độ nhạy: loại các nghiên cứu nguy cơ sai lệch cao",
        "rob_none": "Chưa có nghiên cứu nào được đánh dấu High risk of bias (Dữ liệu nghiên cứu → Đặc điểm nghiên cứu).",
        "rob_excluded": "Đã loại: {ids}",
        "loo_title": "Phân tích độ nhạy bỏ lần lượt từng nghiên cứu",
        "loo_treatment": "Hiệu quả cần theo dõi",
        "loo_hint": "Mỗi dòng chạy lại mạng lưới khi bỏ nghiên cứu đó. Dòng cam: bỏ nghiên cứu làm thay đổi việc CI có loại trừ giá trị null hay không, với ít nhất một treatment.",
        "loo_summary": "{treatment}: ước lượng dao động từ {low} đến {high} qua các lần bỏ nghiên cứu (mô hình đầy đủ {full}).",
        "open_file": "Mở",
        "open_hint": "File NMA (.decide / .json) hoặc dữ liệu nghiên cứu (.csv)",
        "import_hint": "Nhập dữ liệu nghiên cứu từ RevMan (.rm5 / .csv) hoặc Covidence (.csv)",
        "import_button": "Nhập từ file",
        "import_done": "Đã nhập {n} nghiên cứu.",
        "load_example": "Nạp mạng lưới ví dụ",
        "save": "Lưu",
        "save_as": "Lưu thành",
        "filename": "Tên file",
        "model_settings": "Thiết lập mô hình",
        "outcome": "Loại outcome",
        "outcome_binary": "Nhị phân (số sự kiện / tổng)",
        "outcome_continuous": "Liên tục (mean, SD, n)",
        "outcome_rate": "Đếm (sự kiện / person-time)",
        "outcome_contrast": "Hiệu quả đã báo cáo (ước lượng + SE hoặc CI)",
        "measure": "Thước đo hiệu quả",
        "effects_model": "Mô hình hiệu ứng",
        "fixed": "Fixed",
        "random": "Random",
        "reference": "Treatment tham chiếu",
        "lower_better": "Giá trị thấp hơn là tốt hơn",
        "lower_better_help": "Bật cho biến cố có hại (vd OR tử vong); tắt khi cao hơn là tốt (vd đáp ứng). Dùng để xếp hạng.",
        "zero_correction": "Hiệu chỉnh ô 0",
        "ci_level": "Mức tin cậy",
        "mcmc": "MCMC (Bayesian)",
        "mcmc_soon": "Đang tắt: đang dùng engine frequentist. Bật công tắc (hoặc chọn Engine = Bayesian trong Thiết lập mô hình) để chỉnh các thiết lập này.",
        "mcmc_use": "Dùng engine Bayesian",
        "report_title_ui": "Xuất báo cáo",
        "report_needs_run": "Chạy NMA trước rồi mới xuất báo cáo Word / Excel.",
        "report_build": "Chuẩn bị báo cáo (Word + Excel)",
        "report_building": "Đang tạo báo cáo…",
        "report_stale": "Kết quả đã đổi sau khi tạo báo cáo - bấm chuẩn bị lại.",
        "report_word": "Tải báo cáo Word (.docx)",
        "report_excel": "Tải bảng Excel (.xlsx)",
        "rep_report_title": "Báo cáo network meta-analysis",
        "rep_generated": "Tạo bởi DECIDE NMA Editor ngày {date}",
        "rep_sec_methods": "Phương pháp",
        "rep_sec_network": "Cấu trúc mạng lưới",
        "rep_sec_characteristics": "Đặc điểm nghiên cứu",
        "rep_sec_results": "Hiệu quả tương đối",
        "rep_sec_league": "League table",
        "rep_sec_ranking": "Xếp hạng điều trị",
        "rep_sec_heterogeneity": "Dị biệt",
        "rep_sec_inconsistency": "Inconsistency",
        "rep_sec_cinema": "Độ tin cậy của bằng chứng (CINeMA)",
        "rep_sec_contribution": "Contribution matrix",
        "rep_sec_absolute": "Hiệu quả tuyệt đối",
        "rep_sec_fit": "Độ phù hợp mô hình (Bayesian)",
        "rep_sec_prisma": "Checklist báo cáo PRISMA-NMA",
        "rep_tab_prisma": "Bảng 11. Checklist PRISMA-NMA (Hutton 2015): mục 'Reported' đã có trong báo cáo này; các mục còn lại hoàn thiện trong bản thảo.",
        "sof_tab": "Summary of findings",
        "sof_intro": "Bảng Summary of Findings kiểu GRADE cho mạng (Yepes-Nuñez 2019): bằng chứng trực tiếp, hiệu quả tương đối và tuyệt đối, mức tin cậy từ CINeMA và xếp hạng. Bố cục chọn ở đây là bố cục in trong báo cáo Word / Excel.",
        "sof_layout": "Dòng",
        "sof_layout_reference": "Mỗi điều trị so với {ref}",
        "sof_layout_all": "Mọi cặp",
        "sof_layout_comparator": "Mỗi điều trị so với một comparator tự chọn",
        "sof_comparator": "Comparator",
        "rep_sec_sof": "Summary of findings",
        "rep_tab_sof": "Bảng 3b. Summary of findings (kiểu GRADE): bằng chứng trực tiếp, hiệu quả tương đối và tuyệt đối, mức tin cậy (CINeMA) và xếp hạng.",
        "prisma_tab": "PRISMA-NMA",
        "prisma_intro": "Checklist báo cáo theo PRISMA mở rộng cho NMA (Hutton và cs. 2015): 27 mục PRISMA + 5 mục riêng NMA (S1–S5). Editor gợi ý trạng thái cho các mục mà báo cáo này đã bao phủ; chị sửa cột Status và ghi vị trí báo cáo trong bản thảo. Checklist được đưa vào báo cáo Word / Excel.",
        "prisma_summary": "Reported {reported} · Partly {partly} · To do {todo} · Not applicable {na} / {total} mục",
        "prisma_reset": "Về gợi ý ban đầu",
        "prisma_download": "Tải checklist (CSV)",
        "rep_sec_references": "Tài liệu tham khảo",
        "rep_fig_network": "Hình 1. Mạng lưới các nghiên cứu (kích thước node = số bệnh nhân, độ dày cạnh = số nghiên cứu).",
        "rep_fig_forest": "Hình 2. Hiệu quả của từng điều trị so với {ref} ({measure}, khoảng {pct}).",
        "rep_tab_edges": "Bảng 1. Các so sánh trực tiếp trong mạng lưới.",
        "rep_tab_characteristics": "Bảng 2. Đặc điểm nghiên cứu theo so sánh (transitivity).",
        "rep_tab_effects": "Bảng 3. Ước lượng mạng lưới so với {ref}.",
        "rep_tab_league": "Bảng 4. League table ({measure}): điều trị ở cột so với điều trị ở dòng.",
        "rep_tab_ranking": "Bảng 5. Xếp hạng.",
        "rep_tab_nodesplit": "Bảng 6. Node-splitting: bằng chứng trực tiếp so với gián tiếp.",
        "rep_tab_cinema": "Bảng 7. Mức độ tin cậy CINeMA của từng so sánh.",
        "rep_tab_contribution": "Bảng 8. Đóng góp (%) của từng so sánh trực tiếp vào từng ước lượng mạng lưới.",
        "rep_tab_absolute": "Bảng 9. Hiệu quả tuyệt đối trên 1000 bệnh nhân (nguy cơ nền {p0} với {ref}).",
        "rep_tab_fit": "Bảng 10. Độ phù hợp mô hình và hội tụ.",
        "rep_methods_freq": "Network meta-analysis contrast-based được ước lượng bằng bình phương tối thiểu tổng quát (mô hình {effects} effects; phương sai giữa các nghiên cứu τ² theo phương pháp mô-men DerSimonian–Laird tổng quát; thử nghiệm nhiều nhánh được xử lý qua hiệp phương sai trong nghiên cứu). Hiệu quả được báo cáo dưới dạng {measure} với khoảng tin cậy {pct}; khoảng dự đoán random effects dùng phân phối t với {df} bậc tự do. Các điều trị được xếp hạng theo P-score.",
        "rep_methods_bayes_normal": "Network meta-analysis Bayesian được ước lượng với likelihood normal trên thang contrast (mô hình {effects} effects), prior mơ hồ N(0, {prior_sd}²) cho hiệu quả điều trị và prior {tau_prior} cho độ lệch chuẩn giữa các nghiên cứu τ, lấy mẫu hậu nghiệm chính xác (không có vấn đề hội tụ MCMC). Hiệu quả là trung vị hậu nghiệm với khoảng tin cậy Bayesian {pct}; các điều trị được xếp hạng theo SUCRA.",
        "rep_methods_bayes_exact": "Network meta-analysis Bayesian với likelihood arm-level chính xác ({likelihood}) được ước lượng bằng Markov chain Monte Carlo theo NICE DSU TSD 2 (mô hình {effects} effects; {chains} chain × {draws} mẫu sau {warmup} vòng warm-up; prior mơ hồ N(0, {prior_sd}²) cho hiệu quả điều trị và prior {tau_prior} cho τ). Hội tụ được kiểm tra bằng split R-hat và cỡ mẫu hiệu dụng; độ phù hợp mô hình bằng residual deviance, pD và DIC. Hiệu quả là trung vị hậu nghiệm với khoảng tin cậy Bayesian {pct}; các điều trị được xếp hạng theo SUCRA.",
        "rep_methods_inconsistency": "Inconsistency được đánh giá toàn mạng bằng kiểm định design-by-treatment interaction và cục bộ bằng node-splitting ({split_method}).",
        "rep_methods_cinema": "Độ tin cậy của từng ước lượng mạng lưới được xếp mức theo khung CINeMA (sai lệch trong nghiên cứu, sai lệch báo cáo, tính gián tiếp, độ chính xác, dị biệt, không nhất quán), với contribution matrix làm trọng số cho các đánh giá ở mức nghiên cứu.",
        "rep_methods_software": "Phân tích được thực hiện trên DECIDE NMA Editor; engine frequentist đã được đối chiếu với gói R netmeta.",
        "rep_het_text": "τ² = {tau2}, Q = {q} với {df} bậc tự do (p = {p}), I² = {i2}.",
        "rep_dbt_text": "Kiểm định design-by-treatment interaction: Q = {q} với {df} bậc tự do, p = {p}.",
        "rep_ume_text": "DIC mô hình consistency {dic_c} so với mô hình unrelated mean effects {dic_u} ({verdict}).",
        "rep_ume_ok": "không có dấu hiệu inconsistency",
        "rep_ume_bad": "mô hình UME phù hợp hơn: có dấu hiệu inconsistency",
        "rep_no_data": "Chưa chạy.",
        "rep_study": "Nghiên cứu",
        "rep_treatment": "Điều trị",
        "rep_comparison": "So sánh",
        "rep_studies": "Số NC",
        "rep_patients": "Bệnh nhân",
        "rep_estimate": "Ước lượng",
        "rep_ci": "KTC",
        "rep_cri": "CrI",
        "rep_pi": "Khoảng dự đoán",
        "rep_score": "P-score",
        "rep_sucra": "SUCRA",
        "rep_rank": "Hạng trung bình",
        "rep_direct": "Trực tiếp",
        "rep_indirect": "Gián tiếp",
        "rep_difference": "Chênh lệch",
        "rep_confidence": "Độ tin cậy",
        "rep_settings": "Thiết lập",
        "rep_sheet_data": "Dữ liệu nghiên cứu",
        "split_method": "Phương pháp node-splitting",
        "split_side": "SIDE (Dias 2010, NICE TSD 4)",
        "split_sidde": "SIDDE (netmeta)",
        "split_method_help": "SIDE: tham số trực tiếp gắn vào so sánh trong từng nghiên cứu có so sánh đó, nên các nhánh khác của nghiên cứu nhiều nhánh vẫn góp vào ước lượng gián tiếp. SIDDE: toàn bộ nghiên cứu (thiết kế) có so sánh đó là bằng chứng trực tiếp, ước lượng gián tiếp tính ngược từ ước lượng mạng lưới - cách của R netmeta (netsplit). Hai cách trùng nhau khi không có nghiên cứu nhiều nhánh nào chứa so sánh.",
        "chip_random": "random effects",
        "chip_fixed": "fixed effect",
        "chip_reference": "tham chiếu",
        "chip_dres": "Residual deviance {dres:.1f} / {n} điểm dữ liệu",
        "ee_empirical": "Đang dùng engine Bayesian: {n} mẫu hậu nghiệm (rút chung) cho mỗi hiệu quả cũng sẽ được thêm, và PSA của mô hình EE sẽ lấy mẫu trực tiếp từ đó (đúng hình dạng hậu nghiệm và tương quan) thay vì xấp xỉ Lognormal.",
        "colh_parameter": "Tham số",
        "colh_ci_lower": "CI dưới",
        "colh_ci_upper": "CI trên",
        "colh_cri_lower": "CrI dưới",
        "colh_cri_upper": "CrI trên",
        "colh_model": "Mô hình",
        "colh_vs": "so với",
        "colh_estimate_scale": "Ước lượng (thang phân tích)",
        "colh_pi_lower": "PI dưới",
        "colh_pi_upper": "PI trên",
        "colh_variance": "Phương sai",
        "colh_study_removed": "Nghiên cứu bị bỏ",
        "colh_conclusion": "Kết luận thay đổi với",
        "colh_design": "Thiết kế",
        "colh_study_ids": "Mã nghiên cứu",
        "colh_year": "Năm",
        "colh_mean_age": "Tuổi trung bình",
        "colh_female": "% nữ",
        "colh_follow_up": "Theo dõi (tháng)",
        "colh_ratio_id": "Tỷ số (gián tiếp / trực tiếp)",
        "colh_beta_for": "β cho",
        "colh_network": "Mạng lưới",
        "colh_all_studies": "Tất cả NC",
        "colh_without_rob": "Bỏ NC RoB cao",
        "colh_base": "Cơ sở",
        "colh_ume": "UME",
        "colh_nma": "NMA",
        "warn_zero_cell": "{id}: có ô 0 - đã cộng {x} vào mọi ô của nghiên cứu này",
        "warn_excluded_zero": "{id}: bị loại - mọi nhánh đều 0 sự kiện (hoặc toàn sự kiện), nên không có thông tin về tỷ số",
        "warn_zero_variance": "{id}: bị loại - có nhánh phương sai bằng 0 (nguy cơ 0% hoặc 100% mà không hiệu chỉnh ô 0)",
        "warn_no_df": "Không còn bậc tự do cho dị biệt (số so sánh bằng số hiệu quả cần ước lượng): τ² đặt bằng 0, random effects = fixed effect",
        "warn_funnel_points": "Chỉ có {n} điểm: funnel plot và kiểm định Egger có lực kiểm định thấp khi dưới khoảng 10 điểm.",
        "warn_no_residual_df": "Không còn bậc tự do dư: hồi quy khớp hoàn toàn với dữ liệu.",
        "warn_no_rob": "Chưa nhập đánh giá nguy cơ sai lệch (Đặc điểm nghiên cứu): nghiên cứu chưa đánh giá được tính là 'Some concerns'.",
        "warn_low_acceptance": "Tỷ lệ chấp nhận MCMC thấp ở {which} - kiểm tra R-hat / ESS và cân nhắc tăng số draws.",
        "paste_label": "📋 Dán bảng",
        "paste_hint": "Copy các ô trong Excel / Google Sheets (có hoặc không có dòng tiêu đề), dán vào ô dưới rồi bấm Thay toàn bộ hoặc Thêm dòng. Cũng có thể dán thẳng một khối vào một ô của bảng: dữ liệu sẽ điền sang phải và xuống dưới.",
        "paste_replace": "Thay toàn bộ",
        "paste_append": "Thêm dòng",
        "paste_close": "Đóng",
        "paste_empty": "Không có gì để dán.",
        "autosave_status": "Đã tự lưu lúc {time} (autosave/nma-autosave.json)",
        "autosave_found": "Tìm thấy dữ liệu chưa lưu từ lần làm việc trước",
        "autosave_restore": "Khôi phục",
        "autosave_discard": "Bỏ qua",
        "mcmc_running": "MCMC: mô hình {effects} effects…",
        "fit_fe_re": "Fixed so với random effects: mô hình {better} phù hợp hơn theo DIC (chênh lệch {gap:.1f}; dưới 3 thường ưu tiên mô hình fixed effect đơn giản hơn, trừ khi có lý do chờ đợi dị biệt).",
        "bayes_incons_title": "Kiểm tra inconsistency kiểu Bayesian (TSD 4)",
        "bayes_incons_needs_exact": "Có khi dùng engine Bayesian với likelihood chính xác - chạy NMA với Likelihood = Chính xác.",
        "bayes_split_title": "Node-splitting Bayesian",
        "bayes_split_help": "Mỗi so sánh có cả bằng chứng trực tiếp và gián tiếp chạy một lượt MCMC: các nghiên cứu trực tiếp có tham số riêng, phần còn lại của mạng cho ước lượng gián tiếp. p = 2 × min(P(trực tiếp > gián tiếp), P(trực tiếp < gián tiếp)). Nghiên cứu nhiều nhánh dùng chung nhánh nền, nên mâu thuẫn ở một so sánh có thể kéo ước lượng trực tiếp của so sánh khác.",
        "bayes_split_run": "Chạy node-splitting Bayesian ({n} so sánh)",
        "bayes_split_hint": "Dòng cam: p < 0.05. DIC của từng mô hình tách so với {dic} của mô hình consistency - DIC thấp hơn rõ cũng là dấu hiệu inconsistency.",
        "incons_col_direct_n": "Số NC trực tiếp",
        "incons_col_indirect": "Gián tiếp",
        "incons_col_ratio": "Tỷ số trực tiếp / gián tiếp",
        "incons_col_difference": "Chênh lệch trực tiếp − gián tiếp",
        "likelihood": "Likelihood",
        "likelihood_help": "Chính xác: likelihood binomial (OR link logit / RR link log / RD link identity), Poisson (rate ratio) hoặc normal (MD; SMD trên trung bình chuẩn hoá) trên dữ liệu arm-level, bằng MCMC - mô hình của NICE DSU TSD 2; xử lý ô 0 không cần hiệu chỉnh. Xấp xỉ normal: likelihood normal trên contrast, lấy mẫu chính xác trong 1–2 giây.",
        "lik_exact": "Chính xác (arm-level, MCMC)",
        "lik_normal": "Xấp xỉ normal (nhanh)",
        "mcmc_note_exact": "MCMC với likelihood arm-level chính xác (TSD 2): Metropolis-Hastings với đề xuất Laplace cho từng nghiên cứu, cập nhật Gibbs cho hiệu quả, slice sampling cho τ. Các chain chạy song song; bỏ các vòng 'Warm-up' đầu. Kiểm tra R-hat ≤ 1.01 và ESS trong Dashboard. Nghiên cứu contrast-level dùng likelihood normal. Mất vài giây đến một phút.",
        "lik_chip_exact": "likelihood {kind} chính xác",
        "lik_chip_normal": "xấp xỉ normal",
        "fit_title": "Độ phù hợp mô hình (TSD 2): residual deviance và leverage",
        "fit_hint": "Residual deviance trung bình hậu nghiệm {dres:.1f} cho {n} điểm dữ liệu (mô hình phù hợp tốt có Dres gần bằng số điểm dữ liệu), pD {pd:.1f}, DIC {dic:.1f}.",
        "fit_acceptance": "Tỷ lệ chấp nhận MCMC",
        "fit_x": "dấu(r) × √(residual deviance)",
        "fit_y": "Leverage",
        "conv_title": "Biểu đồ hội tụ",
        "conv_help": "Trace: giá trị lấy mẫu của từng chain qua các vòng giữ lại - các chain phải chồng lên nhau và trông như nhiễu dừng. Density: hậu nghiệm của từng chain - phải trùng nhau. Running mean: phải ổn định nhanh. Tham số có R-hat lớn nhất được chọn sẵn.",
        "conv_parameter": "Tham số",
        "conv_chain": "chain",
        "conv_iteration": "Vòng lặp (sau warm-up)",
        "conv_trace": "Trace",
        "conv_density": "Mật độ hậu nghiệm từng chain",
        "conv_running": "Trung bình tích luỹ",
        "conv_summary": "Trung vị hậu nghiệm {median:.4g} ({lower:.4g} đến {upper:.4g}) · R-hat {rhat:.3f} · ESS {ess:.0f}",
        "conv_warning": "Kiểm tra {params}: R-hat > 1.01 hoặc ESS < 400. Tăng số mẫu / warm-up hoặc kiểm tra dữ liệu.",
        "rep_fig_trace": "Hình 3. Biểu đồ trace MCMC (mỗi đường một chain, đã bỏ warm-up) kèm mật độ hậu nghiệm biên của từng tham số.",
        "fit_leverage_note": "Mỗi điểm là một nhánh (hoặc một nghiên cứu contrast-level). Điểm nằm ngoài đường cong ngoài cùng (x² + leverage = 3) phù hợp kém và/hoặc ảnh hưởng mạnh - kiểm tra lại số liệu.",
        "fit_outliers": "Điểm dữ liệu phù hợp kém hoặc ảnh hưởng mạnh: {points}.",
        "ume_title": "Inconsistency: mô hình consistency so với unrelated mean effects (UME) (TSD 4)",
        "ume_help": "Mô hình UME cho mỗi so sánh quan sát được một hiệu quả riêng, không giả định consistency. Nếu UME phù hợp rõ hơn (DIC thấp hơn, điểm nằm dưới đường chéo) thì bằng chứng trực tiếp và gián tiếp mâu thuẫn.",
        "ume_run": "Chạy mô hình UME",
        "ume_consistency": "Consistency (NMA)",
        "ume_model": "Mô hình",
        "ume_points": "Điểm dữ liệu",
        "ume_worse": "DIC thấp hơn {diff:.1f} với mô hình UME: có dấu hiệu inconsistency - xem các điểm nằm dưới đường chéo và các so sánh trong bảng.",
        "ume_ok": "Mô hình consistency phù hợp ngang mô hình UME (chênh lệch DIC {diff:.1f} ≤ 3): không có dấu hiệu inconsistency.",
        "ume_x": "Residual deviance, mô hình consistency",
        "ume_y": "Residual deviance, mô hình UME",
        "ume_devdev_note": "Mỗi điểm là một điểm dữ liệu; điểm nằm rõ dưới đường chéo phù hợp hơn khi bỏ giả định consistency - chỉ ra chỗ có inconsistency.",
        "conv_title": "Công cụ chuyển đổi số liệu (median/IQR, CI, p-value…)",
        "conv_tool": "Bạn cần chuyển đổi gì?",
        "conv_median": "Mean và SD từ median (kèm IQR và/hoặc khoảng min–max)",
        "conv_se": "SE của hiệu quả từ CI hoặc p-value",
        "conv_sd": "SD của một nhánh từ SE hoặc CI",
        "conv_combine": "Gộp hai nhánh cùng một treatment",
        "conv_n": "n",
        "conv_median_v": "Median",
        "conv_mean": "Mean",
        "conv_ratio": "Thước đo tỷ số (OR, RR, HR…) - tính trên thang log",
        "conv_from": "Từ",
        "conv_ci": "CI",
        "conv_p": "p-value",
        "conv_estimate": "Ước lượng",
        "conv_lower": "Cận dưới",
        "conv_upper": "Cận trên",
        "conv_arm": "Nhánh",
        "conv_note_median": "Giả định phân phối gần đối xứng; dữ liệu lệch nhiều thì ước lượng kém tin cậy. Ghi lại cách chuyển đổi ở cột Source.",
        "conv_copy": "Nhập estimate + SE vào bảng Contrast-level (thang log với tỷ số).",
        "conv_fill_title": "Điền vào bảng",
        "conv_fill_target": "Dòng đích",
        "conv_fill_new": "+ Dòng mới (điền Study / Treatment sau)",
        "conv_fill_overwrite": "Ghi đè giá trị đang có",
        "conv_fill_button": "Điền vào bảng",
        "conv_fill_existing": "Dòng này đã có {values}. Tích 'Ghi đè' để thay.",
        "conv_fill_done": "Đã điền {what} vào dòng {row}.",
        "conv_combine_binary": "Biến nhị phân: chỉ cần cộng số sự kiện và số bệnh nhân của hai nhánh.",
        "rank_tab": "Xếp hạng nhiều outcome",
        "rank_intro": "P-score của từng treatment cho mọi phân tích trong file (frequentist, theo mô hình hiệu ứng và thiết lập 'thấp hơn là tốt hơn' của từng phân tích). Xanh = xếp hạng cao, đỏ = thấp; – = không có trong mạng lưới đó.",
        "rank_single": "Thêm phân tích thứ hai (thanh bên → Outcome / phân tích) để so sánh xếp hạng giữa các outcome.",
        "rank_treatment": "Treatment",
        "rank_x": "Trục X",
        "rank_y": "Trục Y",
        "rank_scatter": "Treatment ở góc trên bên phải xếp hạng tốt ở cả hai outcome. P-score không phản ánh độ lớn chênh lệch - nên đọc cùng các ước lượng hiệu quả.",
        "rank_failed": "Không đưa vào - {title}: {error}",
        "prior_title": "Độ nhạy theo prior của τ (Bayesian, random effects)",
        "prior_help": "Chạy lại mô hình random effects với 4 prior khác nhau cho SD giữa các nghiên cứu τ. Kết quả thay đổi nhiều theo prior nghĩa là prior quyết định hơn là dữ liệu - cần báo cáo.",
        "prior_needs": "Chọn engine Bayesian để chạy phần này.",
        "prior_run": "Chạy độ nhạy prior",
        "prior_col": "Prior cho τ",
        "prior_current": "(đang dùng)",
        "abs_title": "Hiệu quả tuyệt đối và NNT",
        "abs_baseline": "Nguy cơ với {ref} (%)",
        "abs_hint": "Nguy cơ của outcome với treatment tham chiếu trong khung thời gian của nghiên cứu - nên dùng quần thể của bài toán quyết định nếu có.",
        "abs_observed": "Nguy cơ gộp ở các nhánh tham chiếu của các thử nghiệm này: {pct}.",
        "abs_hr_note": "Với HR / rate ratio: nguy cơ tại thời điểm quan tâm, giả định proportional hazards.",
        "abs_col_treatment": "So sánh",
        "abs_col_ref": "Với tham chiếu (trên 1000)",
        "abs_col_risk": "Với treatment (trên 1000, CI)",
        "abs_col_diff": "Chênh lệch (trên 1000, CI)",
        "abs_nnt_note": "NNTB = số người cần điều trị để thêm một người có lợi; NNTH = để thêm một người bị hại (làm tròn lên). '∞' khi CI bao gồm không có hiệu quả (Altman 1998).",
        "bayes_not_run": "Đã chọn engine Bayesian nhưng chưa chạy: bấm Chạy NMA để có SUCRA và xác suất xếp hạng. Tạm thời đang hiện kết quả frequentist (P-score).",
        "sucra_col_treatment": "Treatment",
        "sucra_col_mean_rank": "Hạng trung bình",
        "sucra_col_best": "P(tốt nhất)",
        "mcmc_note": "Lấy mẫu hậu nghiệm chính xác (τ trên lưới mịn, hiệu quả rút chính xác theo τ - phương pháp 'DIRECT' của bayesmeta): không cần burn-in, không tự tương quan; các chain là các luồng độc lập để kiểm tra R-hat / ESS. Likelihood trong nghiên cứu là normal trên thang contrast (giống engine frequentist), chưa phải binomial chính xác như NICE TSD 2.",
        "engine": "Engine",
        "frequentist": "Frequentist",
        "bayesian": "Bayesian",
        "tau_prior": "Prior cho τ",
        "tau_informative": "Có thông tin (Turner 2015 / Rhodes 2015)",
        "tau_prior_help": "Uniform / half-normal: mơ hồ. Có thông tin: phân phối dự đoán thực nghiệm cho τ² (log-normal, Turner 2015 cho outcome nhị phân trên thang log OR; log-t, Rhodes 2015 cho outcome liên tục trên thang SMD) chọn theo loại outcome và loại so sánh – khuyến nghị cho mạng nhỏ, khi prior mơ hồ cho khoảng rất rộng.",
        "tau_outcome_type": "Loại outcome",
        "tau_comparison_type": "Loại so sánh",
        "tau_general_used": "phân phối chung",
        "tau_cell_missing": "Ô outcome × so sánh này chưa được nhập – dùng phân phối chung. Các ô đọc từ priors/tau_priors.json.",
        "class_model": "Class của thuốc",
        "class_none": "Không dùng class effect",
        "class_common": "Class effect chung (common)",
        "class_exchangeable": "Class effect trao đổi được (exchangeable)",
        "class_model_help": "Dùng cột Class của bảng Treatments (class có từ hai thuốc trở lên trong mạng; chỉ với engine Bayesian likelihood chính xác). Common: mọi thuốc trong class có chung một hiệu quả. Exchangeable: hiệu quả từng thuốc lấy từ phân phối của class N(trung bình class, σ_class²), thuốc ít dữ liệu được kéo về phía class (NICE DSU TSD 3).",
        "class_sd_scale": "Prior σ_class: scale half-normal",
        "class_unavailable": "Class effect cần engine Bayesian likelihood chính xác và một class có từ hai thuốc trở lên (bảng Treatments, cột Class).",
        "class_title": "Class effect",
        "class_common_help": "Các thuốc trong class có chung một hiệu quả so với reference; SUCRA xếp hạng các class.",
        "class_exchangeable_help": "Hiệu quả trung bình của class so với reference; từng thuốc bị kéo về phía đó. SUCRA xếp hạng các class.",
        "class_sd_text": "Độ lệch chuẩn giữa các thuốc trong một class (σ_class): trung vị {median:.3f} (CrI {lower:.3f} đến {upper:.3f}). Giá trị nhỏ nghĩa là các thuốc trong class hoạt động giống nhau.",
        "class_compare_button": "Chạy mô hình không class để so sánh DIC",
        "class_compare_text": "DIC có class effect {dic_c:.1f} so với không có {dic_n:.1f} (chênh {diff:+.1f}); pD {pd_c:.1f} so với {pd_n:.1f}. DIC thấp hơn là tốt hơn; chênh dưới 3 chưa đáng kể.",
        "class_not_used": "Đã yêu cầu class effect nhưng không dùng được:",
        "rep_sec_class": "Class effect",
        "rep_tab_class": "Bảng 10b. Class effect so với {ref} ({measure}, khoảng tin cậy Bayesian {pct}) và SUCRA giữa các class.",
        "thr_tab": "Phân tích ngưỡng",
        "thr_intro": "Mỗi bằng chứng có thể thay đổi bao nhiêu trước khi điều trị tốt nhất đổi? Từng contrast quan sát được dịch chuyển riêng lẻ (các dữ liệu khác giữ nguyên); khoảng bất biến là vùng mà khuyến nghị không đổi (Phillippo 2018). Dựa trên mô hình GLS tần suất (tau² giữ cố định); điểm có giới hạn nằm trong khoảng tin cậy 95% của chính dữ liệu là điểm dễ vỡ.",
        "thr_basis": "Cơ sở ra quyết định",
        "thr_basis_effect": "Hiệu quả điều trị",
        "thr_basis_nmb": "Lợi ích tiền tệ ròng (CEA/CUA)",
        "thr_nmb_help": "Nhập chi phí và hiệu quả của từng điều trị tại ước lượng NMA hiện tại, cùng độ nhạy: hiệu quả và chi phí thay đổi bao nhiêu khi hiệu quả NMA của nó tăng 1 đơn vị (độ nhạy cục bộ từ mô hình kinh tế, ví dụ chạy lại mô hình với hiệu quả thay đổi nhẹ). Bảng ICER chỉ để tham khảo: tại mọi ngưỡng, ICER của điều trị mới tốt nhất so với điều trị cũ bằng đúng mức sẵn sàng chi trả.",
        "thr_wtp": "Mức sẵn sàng chi trả trên mỗi đơn vị hiệu quả",
        "thr_col_treatment": "Điều trị",
        "thr_col_cost": "Chi phí",
        "thr_col_effect": "Hiệu quả (tuyệt đối)",
        "thr_col_d_effect": "dHiệu quả / dd",
        "thr_col_d_cost": "dChi phí / dd",
        "thr_col_d_effect_help": "Thay đổi hiệu quả tuyệt đối của điều trị này trên mỗi đơn vị hiệu quả NMA (theo thang phân tích)",
        "thr_col_d_cost_help": "Thay đổi chi phí của điều trị này trên mỗi đơn vị hiệu quả NMA (theo thang phân tích)",
        "thr_best": "Điều trị tốt nhất hiện tại: **{best}** (thứ tự: {order}).",
        "thr_icer_title": "Hiệu quả chi phí so với điều trị tham chiếu",
        "thr_icer_help": "ICER = chi phí gia tăng / hiệu quả gia tăng so với điều trị tham chiếu.",
        "thr_inc_cost": "Chi phí gia tăng",
        "thr_inc_effect": "Hiệu quả gia tăng",
        "thr_scope": "Dịch chuyển",
        "thr_scope_studies": "Từng contrast của nghiên cứu",
        "thr_scope_network": "Từng ước lượng mạng",
        "thr_study": "Nghiên cứu",
        "thr_comparison": "So sánh",
        "thr_estimate": "Ước lượng",
        "thr_lower": "Giới hạn dưới",
        "thr_lower_best": "Điều trị tốt nhất khi thấp hơn",
        "thr_upper": "Giới hạn trên",
        "thr_upper_best": "Điều trị tốt nhất khi cao hơn",
        "thr_fragile": "Nằm trong khoảng 95% của dữ liệu",
        "thr_never": "không bao giờ",
        "thr_download": "Tải bảng ngưỡng (CSV)",
        "thr_fragile_yes": "Dễ vỡ",
        "thr_fragile_no": "Vững",
        "thr_axis": "Ước lượng (thang tự nhiên)",
        "thr_chart_help": "Chấm = ước lượng quan sát; đường = khoảng bất biến (khuyến nghị không đổi trong khoảng này). Đỏ = giới hạn nằm trong khoảng 95% của ước lượng.",
        "yes": "có",
        "no": "không",
        "rep_sec_threshold": "Phân tích ngưỡng",
        "rep_tab_threshold": "Bảng 10c. Phân tích ngưỡng ({basis}): với mỗi contrast của nghiên cứu, giá trị tại đó điều trị tốt nhất (hiện là {best}) thay đổi, cùng điều trị thay thế.",
        "rep_methods_threshold": "Phân tích ngưỡng (Phillippo và cs. 2018) được thực hiện trên cơ sở {basis}: mỗi contrast quan sát được dịch chuyển riêng lẻ, các dữ liệu khác giữ nguyên, và giới hạn của khoảng bất biến của nó, vượt qua đó điều trị tốt nhất thay đổi, được tính từ các ước lượng mạng vốn tuyến tính theo dữ liệu.",
        "rep_basis_effect": "hiệu quả điều trị",
        "rep_basis_nmb": "lợi ích tiền tệ ròng",
        "cmp_tab": "NMA theo thành phần",
        "cmp_intro": "NMA thành phần cộng tính (Rücker 2020): mỗi điều trị là một tập các thành phần và hiệu quả bằng tổng hiệu quả các thành phần. Nhập thành phần của từng điều trị, ngăn cách bằng +; để trống thì điều trị đó là một thành phần riêng (điều trị tham chiếu là tập rỗng).",
        "cmp_col_components": "Thành phần",
        "cmp_col_help": "ví dụ CBT + thuốc",
        "cmp_component": "Thành phần",
        "cmp_none": "Nhập thành phần cho ít nhất một điều trị để khớp mô hình cộng tính.",
        "cmp_effects_title": "Hiệu quả của từng thành phần",
        "cmp_treatments_title": "Điều trị: mô hình cộng tính so với NMA chuẩn",
        "cmp_additive": "Mô hình cộng tính",
        "cmp_standard": "NMA chuẩn",
        "cmp_additivity": "Kiểm định tính cộng tính: Q = {q}, {df} bậc tự do, p = {p} (p nhỏ gợi ý các thành phần tương tác). Chỉ tần suất.",
        "rep_sec_components": "Phân tích gộp mạng theo thành phần",
        "rep_tab_components": "Bảng 10d. NMA thành phần cộng tính: hiệu quả của từng thành phần ({measure}) và hiệu quả điều trị so với {ref}; kiểm định tính cộng tính Q = {q}, {df} bậc tự do (p = {p}).",
        "rep_methods_components": "NMA thành phần cộng tính (Rücker và cs. 2020) được khớp: hiệu quả của một điều trị so với điều trị tham chiếu bằng tổng hiệu quả các thành phần của nó. Tính cộng tính được đánh giá bằng cách so sánh mô hình cộng tính với NMA chuẩn.",
        "des_tab": "Bằng chứng không ngẫu nhiên",
        "des_intro": "Đánh dấu các nghiên cứu không ngẫu nhiên hóa. Mạng được ước lượng lại khi loại chúng, khi giữ nguyên trọng số và khi giảm trọng số theo alpha (phương sai của chúng chia cho alpha), để thấy kết luận phụ thuộc vào chúng bao nhiêu. Đây là phân tích độ nhạy; phân tích chính không đổi. Mô hình tần suất.",
        "des_col_nrs": "Không ngẫu nhiên",
        "des_none": "Đánh dấu ít nhất một nghiên cứu không ngẫu nhiên.",
        "des_alphas": "Hệ số giảm trọng số (alpha)",
        "des_note": "Mỗi ô là hiệu quả so với tham chiếu (khoảng tin cậy). alpha = 0,25 nghĩa là nghiên cứu không ngẫu nhiên chỉ tính bằng một phần tư cỡ mẫu của nó.",
        "rep_sec_design": "Bằng chứng không ngẫu nhiên",
        "rep_tab_design": "Bảng 10e. Độ nhạy với các nghiên cứu không ngẫu nhiên ({nrs}): hiệu quả so với {ref} ({measure}, khoảng tin cậy {pct}) khi loại chúng, giữ nguyên trọng số và giảm trọng số.",
        "rep_methods_design": "Các nghiên cứu được đánh dấu không ngẫu nhiên được xử lý trong phân tích độ nhạy: loại bỏ, giữ nguyên trọng số và giảm trọng số bằng cách chia phương sai cho hệ số alpha.",
        "tv_tab": "HR theo thời gian",
        "tv_intro": "Hazard ratio theo khoảng thời gian (ví dụ 0-6, 6-12, 12+ tháng). Nhập mỗi dòng một study, một khoảng và một contrast, với log HR và SE, hoặc HR và KTC 95%. Mỗi khoảng chạy một NMA riêng, coi các khoảng của cùng một study là độc lập (xấp xỉ: chúng dùng chung bệnh nhân). Study nhiều nhánh báo hai contrast trong một khoảng cần thêm SE nền.",
        "tv_from": "Từ",
        "tv_to": "Đến",
        "tv_lower": "HR cận dưới 95%",
        "tv_upper": "HR cận trên 95%",
        "tv_baseline_se": "SE nền",
        "tv_baseline_help": "SE của log hazard ở nhánh đối chứng chung; chỉ cho study nhiều nhánh",
        "tv_none": "Nhập ít nhất một dòng đầy đủ (study, treatment, versus, từ, đến, và dữ liệu HR).",
        "tv_reference": "Điều trị tham chiếu",
        "tv_unit": "Đơn vị thời gian",
        "tv_results": "Hazard ratio so với tham chiếu, theo khoảng",
        "tv_chart_help": "Mỗi thanh trải dài một khoảng thời gian tại HR của mạng (trục log).",
        "tv_ph_title": "Kiểm tra tỉ lệ nguy cơ (PH)",
        "tv_ph_help": "Q Cochran của log HR các khoảng (giả định độc lập). p nhỏ gợi ý HR thay đổi theo thời gian; p lớn không chứng minh PH (công suất thấp).",
        "tv_level": "Mức",
        "tv_item": "Contrast / điều trị",
        "tv_intervals": "Số khoảng",
        "tv_flag": "HR thay đổi theo thời gian?",
        "tv_possibly": "có thể",
        "tv_no_evidence": "không có bằng chứng",
        "rep_sec_piecewise": "Hazard ratio theo thời gian",
        "rep_tab_piecewise": "Bảng 10f. Hazard ratio theo khoảng thời gian so với {ref} (khoảng tin cậy {pct}), từ một NMA riêng cho mỗi khoảng; các khoảng của cùng một study được coi là độc lập.",
        "rep_methods_piecewise": "Hazard ratio được phân tích theo khoảng thời gian: một NMA dựa trên contrast được khớp riêng cho hazard ratio của từng khoảng, coi các khoảng của một study là độc lập. Tỉ lệ nguy cơ được đánh giá bằng Q Cochran của log hazard ratio các khoảng.",
        "sv_tab": "NMA sống còn",
        "sv_intro": "NMA cho đường cong sống còn: hazard exponential, Weibull, Gompertz hoặc fractional polynomial khớp với số sự kiện và người-thời-gian theo khoảng (likelihood Poisson, Bayesian, hiệu quả điều trị trên mọi hệ số hazard). Số liệu có thể nhập tay hoặc dựng lại từ đường Kaplan-Meier đã công bố (Guyot 2012). So sánh mô hình bằng DIC, kiểm tra đường cong so với thời gian theo dõi, rồi xuất sang mô hình kinh tế.",
        "sv_data_title": "Dữ liệu theo khoảng",
        "sv_data_help": "Mỗi dòng là một study, một nhánh, một khoảng thời gian: số sự kiện và người-thời-gian trong khoảng (cùng đơn vị thời gian với giới hạn khoảng).",
        "sv_km_title": "Dựng lại dữ liệu theo khoảng từ đường Kaplan-Meier",
        "sv_km_help": "Số hoá từng đường cong đã công bố (thời gian, sống còn) và chép số người còn nguy cơ in dưới hình (dòng đầu là thời điểm 0 với cỡ mẫu nhánh). Dữ liệu được dựng lại bằng thuật toán Guyot rồi điền vào bảng khoảng bên trên. Độ chính xác phụ thuộc điểm số hoá; dùng IPD thật khi có.",
        "sv_km_points": "Điểm trên đường cong",
        "sv_km_risk": "Số người còn nguy cơ",
        "sv_breaks": "Giới hạn các khoảng (bắt đầu từ 0)",
        "sv_breaks_error": "Giới hạn khoảng phải là các số tăng dần bắt đầu từ 0",
        "sv_reconstruct": "Dựng lại và điền vào bảng khoảng",
        "sv_none": "Nhập dữ liệu theo khoảng cho ít nhất hai nhánh (hoặc dựng lại từ đường cong).",
        "sv_time_factor": "Đơn vị thời gian của mô hình tính theo năm",
        "sv_time_factor_help": "Để xuất sang PSM: 1/12 nếu là tháng, 1 nếu là năm",
        "sv_model": "Mô hình",
        "sv_warmup": "Warm-up",
        "sv_draws": "Số mẫu mỗi chain",
        "sv_run": "Chạy mô hình này",
        "sv_run_all": "Chạy exponential, Weibull và Gompertz",
        "sv_running": "Đang lấy mẫu...",
        "sv_compare": "So sánh mô hình (DIC, thấp hơn là tốt hơn)",
        "sv_show": "Xem mô hình",
        "sv_effects_title": "Hiệu quả điều trị trên các hệ số hazard",
        "sv_effects_help": "Mỗi dòng là hiệu quả của một điều trị so với tham chiếu trên một hệ số của log-hazard. Với Weibull, hệ số đầu là log-hazard ở cuối thời gian theo dõi, hệ số sau là thay đổi của log shape.",
        "sv_baseline": "Hazard nền của điều trị tham chiếu",
        "sv_baseline_pooled": "Gộp các study có nhánh tham chiếu",
        "sv_horizon": "Thời gian tối đa của đường cong (đơn vị thời gian)",
        "sv_survival": "Sống còn",
        "sv_chart_help": "Đường trung vị kèm dải tin cậy; đường đứt nét là hết thời gian theo dõi: sau đó là ngoại suy.",
        "sv_export_title": "Xuất sang mô hình kinh tế",
        "sv_export_help": "Bảng sống còn: đường trung vị. Mẫu PSA: mỗi draw là một mẫu hậu nghiệm, thẳng hàng giữa các điều trị. Tham số PSM: tham số dạng đóng cho node PSM (exponential, Weibull, Gompertz).",
        "sv_dl_table": "Đường sống còn (CSV)",
        "sv_psa_n": "Số draw PSA",
        "sv_dl_psa": "Mẫu đường cong PSA (CSV)",
        "sv_dl_psm": "Tham số PSM (CSV)",
        "sv_psm_note": "Đường fractional polynomial không có dạng đóng trong node PSM: dùng bảng sống còn hoặc mẫu PSA.",
        "rep_sec_survival": "NMA sống còn",
        "rep_tab_survival": "Bảng 10g. NMA sống còn: so sánh mô hình (DIC, thấp hơn là tốt hơn) và hiệu quả điều trị trên các hệ số log-hazard của mô hình được chọn ({model}) so với {ref}.",
        "rep_methods_survival": "NMA sống còn được khớp với số sự kiện và người-thời-gian theo khoảng (likelihood Poisson; dữ liệu dựng lại từ đường Kaplan-Meier đã công bố khi có ghi chú) với hazard exponential, Weibull, Gompertz hoặc fractional polynomial (Jansen 2011), hiệu quả điều trị trên mọi hệ số hazard, lấy mẫu Metropolis-adjusted Langevin; các mô hình được so sánh bằng DIC.",
        "pa_tab": "Điều chỉnh quần thể",
        "pa_intro": "Điều chỉnh khác biệt giữa các quần thể bệnh nhân khi bạn có dữ liệu bệnh nhân riêng lẻ (IPD) cho một số thử nghiệm và chỉ có dữ liệu tổng hợp cho các thử nghiệm khác: MAIC gán trọng số cho một thử nghiệm có IPD để khớp đặc điểm của thử nghiệm khác; ML-NMR khớp một hồi quy trên cả dữ liệu riêng lẻ và tổng hợp (Phillippo 2020). Cả hai đều giả định mọi yếu tố điều biến hiệu quả đã được đo; cần kiểm tra bằng ý kiến lâm sàng.",
        "pa_privacy": "IPD chỉ nằm trong phiên làm việc này: không được lưu vào file, autosave hay báo cáo (báo cáo chỉ có bảng tóm tắt).",
        "pa_upload": "File IPD (CSV, mỗi dòng một bệnh nhân)",
        "pa_need_ipd": "Tải file IPD lên để bắt đầu.",
        "pa_loaded": "Đã nạp {rows} bệnh nhân, {cols} cột.",
        "pa_preview": "Xem các dòng đầu",
        "pa_maic_tab": "MAIC",
        "pa_mlnmr_tab": "ML-NMR",
        "pa_arm_col": "Cột nhánh điều trị",
        "pa_outcome_type": "Loại outcome",
        "pa_type_binary": "Nhị phân",
        "pa_type_continuous": "Liên tục",
        "pa_type_survival": "Thời gian đến biến cố",
        "pa_two_arms": "Cột nhánh cần có ít nhất hai giá trị khác nhau.",
        "pa_treatment": "Điều trị",
        "pa_comparator": "Đối chứng",
        "pa_event_col": "Cột biến cố (0/1)",
        "pa_value_col": "Cột outcome",
        "pa_time_col": "Cột thời gian",
        "pa_cov_cols": "Covariate cần khớp (số)",
        "pa_pick_covs": "Chọn ít nhất một covariate.",
        "pa_target": "Quần thể đích (thử nghiệm tổng hợp)",
        "pa_target_help": "Nhập trung bình covariate báo cáo của thử nghiệm tổng hợp (tỉ lệ với covariate 0/1) và, nếu có, độ lệch chuẩn.",
        "pa_match_sd": "Khớp cả độ lệch chuẩn (khi đã nhập)",
        "pa_se_method": "Sai số chuẩn",
        "pa_se_auto": "Robust (bootstrap cho thời gian đến biến cố)",
        "pa_se_boot": "Bootstrap (ước lượng lại trọng số)",
        "pa_run": "Chạy",
        "pa_running": "Đang khớp...",
        "pa_ess": "Cỡ mẫu hiệu dụng",
        "pa_covariate": "Covariate",
        "pa_unweighted": "IPD, chưa gán trọng số",
        "pa_weighted": "IPD, có trọng số",
        "pa_target_col": "Đích",
        "pa_analysis": "Phân tích",
        "pa_unadjusted": "Chưa điều chỉnh",
        "pa_adjusted": "Đã điều chỉnh (MAIC)",
        "pa_balance": "Cân bằng covariate",
        "pa_effect": "Hiệu quả",
        "pa_scale_binary": "tỉ số odds",
        "pa_scale_continuous": "chênh lệch trung bình",
        "pa_scale_survival": "hazard ratio",
        "pa_weight": "Trọng số",
        "pa_anchor_title": "So sánh gián tiếp có neo",
        "pa_anchor_help": "Nếu thử nghiệm tổng hợp so sánh một điều trị khác với cùng đối chứng, nhập hiệu quả và khoảng 95% của nó (cùng thang đo) để có so sánh gián tiếp đã điều chỉnh giữa điều trị của IPD và điều trị đó (Bucher). Không có đối chứng chung thì so sánh không neo chỉ đáng tin nếu mọi yếu tố tiên lượng và điều biến hiệu quả đều được khớp, điều hiếm khi đạt được.",
        "pa_agg_effect": "Hiệu quả của thử nghiệm tổng hợp",
        "pa_agg_lower": "Cận dưới 95%",
        "pa_agg_upper": "Cận trên 95%",
        "pa_agg_positive": "Thang đo tỉ số cần giá trị dương.",
        "pa_anchor_result": "So sánh gián tiếp đã điều chỉnh: {est} (KTC 95% {lo} đến {hi})",
        "pa_ml_intro": "Outcome nhị phân. Nhánh IPD dùng likelihood cá thể; nhánh tổng hợp (số biến cố, n, trung bình và SD của covariate) được tích phân trên phân phối covariate của chúng. Hiệu quả được báo cáo tại một hồ sơ covariate.",
        "pa_study_col": "Cột study",
        "pa_em_cols": "Yếu tố điều biến hiệu quả",
        "pa_agg_title": "Các nhánh dữ liệu tổng hợp",
        "pa_agg_help": "Mỗi dòng một nhánh của thử nghiệm tổng hợp: study, điều trị, số biến cố, n và trung bình covariate (tỉ lệ với covariate 0/1) và SD.",
        "pa_shared": "Yếu tố điều biến dùng chung cho các điều trị",
        "pa_shared_help": "Một hệ số tương tác cho mỗi covariate chung cho mọi điều trị tích cực so với tham chiếu (giả định phổ biến khi dữ liệu thưa); bỏ chọn để mỗi điều trị một hệ số.",
        "pa_profile": "Hồ sơ covariate",
        "pa_parameter": "Tham số",
        "pa_or": "Tỉ số odds so với tham chiếu",
        "pa_ml_effects": "Tỉ số odds có điều kiện tại hồ sơ",
        "pa_coefficients": "Hệ số (thang log-odds, covariate được căn tâm tại trung bình IPD gộp)",
        "pa_ml_note": "Maximum likelihood với intercept riêng từng study và xấp xỉ chuẩn; chưa có phiên bản random-effects hay Bayesian. Nếu không có IPD thì các tương tác chỉ dựa trên biến thiên giữa các study.",
        "rep_sec_popadj": "Điều chỉnh quần thể",
        "rep_tab_maic": "Bảng 10h. So sánh gián tiếp điều chỉnh theo khớp (MAIC): cân bằng covariate trước và sau gán trọng số và hiệu quả đã điều chỉnh (kèm cỡ mẫu hiệu dụng); báo cáo này không chứa dữ liệu bệnh nhân riêng lẻ.",
        "rep_tab_mlnmr": "Bảng 10i. Meta-regression mạng đa cấp (outcome nhị phân, maximum likelihood): hệ số và tỉ số odds có điều kiện so với tham chiếu tại hồ sơ covariate nêu trên.",
        "rep_methods_maic": "So sánh gián tiếp điều chỉnh theo khớp (Signorini 2010; NICE DSU TSD 18) gán trọng số cho dữ liệu bệnh nhân riêng lẻ bằng phương pháp moment để phân phối covariate có trọng số khớp với thử nghiệm tổng hợp; hiệu quả được ước lượng trong mẫu có trọng số với phương pháp tính sai số chuẩn đã nêu.",
        "rep_methods_mlnmr": "Meta-regression mạng đa cấp (Phillippo 2020) kết hợp dữ liệu riêng lẻ và tổng hợp: các nhánh tổng hợp được mô hình hoá bằng cách tích phân mô hình cấp cá thể trên phân phối covariate của chúng (quasi-Monte Carlo), và hiệu quả được ước lượng bằng maximum likelihood.",
        "dr_tab": "Liều - đáp ứng",
        "dr_intro": "NMA liều-đáp ứng (MBNMA): hiệu quả phụ thuộc liều của từng hoạt chất qua một hàm (tuyến tính, Emax, ...), ước lượng từ mọi nhánh; nhánh liều 0 là placebo. Mỗi dòng một nhánh: study, hoạt chất, liều và hoặc trung bình và SE (liên tục) hoặc số biến cố và n (nhị phân, thang log odds). Maximum likelihood, intercept riêng từng study, không có random effect giữa các study.",
        "dr_none": "Nhập ít nhất bốn nhánh (một số study có nhánh liều 0).",
        "dr_groups_title": "Class effect: dùng chung đường cong giữa các hoạt chất",
        "dr_groups_help": "Các hoạt chất cùng tên nhóm dùng chung một bộ tham số liều-đáp ứng (class effect chung). Giữ nguyên nếu muốn mỗi hoạt chất một đường cong.",
        "dr_mixed": "Dùng hoặc trung bình và SE hoặc số biến cố và n cho mọi nhánh, không trộn lẫn.",
        "dr_functions": "Các hàm liều-đáp ứng cần khớp",
        "dr_compare": "So sánh các dạng hàm",
        "dr_function": "Hàm",
        "dr_deviance": "Residual deviance",
        "dr_fit_p": "p độ khớp",
        "dr_dispersion": "Dispersion",
        "dr_param_help": "Emax: theta0 = hiệu quả tối đa, theta1 = log ED50. Hill: theta2 = log hệ số Hill. Exponential: theta1 = log thang liều.",
        "dr_dose": "Liều",
        "dr_effect_axis_continuous": "Hiệu quả so với liều 0 (chênh lệch trung bình)",
        "dr_effect_axis_binary": "Hiệu quả so với liều 0 (log odds ratio)",
        "dr_chart_help": "Đường cong với dải 95%; chấm là hiệu quả quan sát của từng nhánh so với nhánh placebo cùng study; đường đứt nét là liều cao nhất đã nghiên cứu của mỗi hoạt chất, vượt qua đó là ngoại suy.",
        "rep_sec_dose": "NMA liều-đáp ứng",
        "rep_tab_dose": "Bảng 10j. MBNMA liều-đáp ứng: so sánh các dạng hàm (AIC, thấp hơn là tốt hơn) và tham số của hàm được chọn ({function}). Theta là tham số của hàm; ED50 và Hill ở thang log.",
        "rep_methods_dose": "NMA dựa trên mô hình cho liều-đáp ứng (Mawdsley 2016) được khớp trên dữ liệu cấp nhánh bằng maximum likelihood với intercept riêng từng study, nên chỉ các đối chiếu liều trong cùng study cung cấp thông tin cho đường cong; các dạng hàm được so sánh bằng tiêu chí thông tin Akaike.",
        "chip_tau_prior": "Prior τ: {label}",
        "tau_uniform": "Uniform(0, U)",
        "tau_half_normal": "Half-normal(0, s)",
        "tau_upper": "U",
        "tau_scale": "s",
        "prior_sd": "SD prior cho hiệu quả",
        "sucra": "Xếp hạng (SUCRA)",
        "rankogram": "Xác suất xếp hạng (1 = tốt nhất)",
        "bayes_diag": "Tóm tắt hậu nghiệm và hội tụ",
        "bayes_fit": "Độ phù hợp mô hình",
        "bayes_note_results": "Kết quả Bayesian: trung vị hậu nghiệm với khoảng tin cậy (CrI) {pct}. Node-splitting và leave-one-out dùng engine frequentist.",
        "chains": "Chains",
        "seed": "Seed",
        "warmup": "Warm-up",
        "draws": "Draws / chain",
        "treatments": "Treatments",
        "add_row": "+ Thêm dòng",
        "delete_row": "Xoá dòng",
        "undo": "↶ Hoàn tác (⌘Z)",
        "redo": "↷ Làm lại (⇧⌘Z)",
        "treatments_help": "Aliases: các cách viết khác, cách nhau bằng dấu phẩy. Color: mã #hex, dùng khi Màu node = Custom.",
        "study_data": "Dữ liệu nghiên cứu",
        "arm_level": "Arm-level",
        "contrast_level": "Contrast-level (HR / OR / MD)",
        "contrast_help": "Nhập estimate + se theo thang phân tích (log với tỷ số), hoặc effect + lower + upper theo thang gốc. Study nhiều nhánh: cùng một comparator ở mọi dòng và có baseline_se.",
        "network": "Mạng lưới",
        "view": "Hiển thị",
        "node_colour": "Màu node",
        "by_class": "Theo nhóm",
        "single": "Một màu",
        "custom": "Tuỳ chỉnh",
        "size_by": "Kích thước theo",
        "patients": "Bệnh nhân",
        "studies": "Số nghiên cứu",
        "equal": "Bằng nhau",
        "selected_edge": "Xem so sánh",
        "no_edge": "(không)",
        "direct": "Trực tiếp",
        "indirect_via": "Đường gián tiếp qua",
        "none": "không có",
        "network_ok": "Mạng lưới đã kết nối",
        "fix_errors": "Cần sửa trước khi chạy",
        "add_missing": "Thêm các treatment còn thiếu vào danh sách",
        "warnings": "Ghi chú",
        "check_title": "Kiểm tra mạng lưới",
        "results_title": "Kết quả base case",
        "details_title": "Chi tiết mô hình",
        "stale": "Dữ liệu đã thay đổi sau lần chạy này - bấm Chạy NMA lại.",
        "not_run": "Bấm Chạy NMA để ước lượng mô hình.",
        "forest": "Forest plot so với tham chiếu",
        "rankings": "Xếp hạng (P-score)",
        "league": "League table",
        "league_hint": "Ô = hiệu quả của treatment ở cột so với treatment ở dòng; sắp theo P-score.",
        "heterogeneity": "Dị biệt (heterogeneity)",
        "observations": "Các so sánh theo nghiên cứu đưa vào mô hình",
        "excluded": "Nghiên cứu bị loại",
        "download_csv": "Tải CSV",
        "illustrative": "Dữ liệu ví dụ chỉ để minh hoạ, không phải kết quả thử nghiệm thật.",
        "parse_error": "Không đọc được dữ liệu",
        "cinema_tab": "Độ tin cậy (CINeMA)",
        "contrib_tab": "Contribution matrix",
        "cinema_intro": "Độ tin cậy của từng ước lượng mạng lưới (CINeMA / GRADE-NMA): 6 tiêu chí được xếp mức tự động từ các kết quả đã có. Rê chuột lên ô để xem lý do; * = đã sửa tay.",
        "cinema_margin_ratio": "Hiệu quả có ý nghĩa lâm sàng ({measure})",
        "cinema_margin_ratio_help": "Khoảng tương đương 1/m đến m, vd 1.25 → 0.80–1.25. Dùng cho imprecision và heterogeneity.",
        "cinema_margin_diff": "Chênh lệch có ý nghĩa lâm sàng ({measure})",
        "cinema_margin_diff_help": "Khoảng tương đương −m đến m. 0 = không đặt ngưỡng: chỉ so với đường không hiệu quả.",
        "cinema_rule": "Quy tắc cho RoB / indirectness",
        "cinema_rule_average": "Trung bình",
        "cinema_rule_majority": "Đa số",
        "cinema_rule_help": "Trung bình: mức của từng nghiên cứu (Low 1, Some 2, High 3) có trọng số theo đóng góp vào ước lượng (< 1.5 không đáng lo, < 2.5 some, còn lại major). Đa số: mức có đóng góp lớn nhất.",
        "cinema_needs_run": "Bấm Chạy NMA trước - CINeMA dùng kết quả NMA, inconsistency và funnel plot.",
        "cinema_freq_note": "CINeMA và contribution matrix dùng kết quả frequentist.",
        "cinema_edit": "Xem lại hoặc sửa đánh giá",
        "cinema_comparison": "So sánh",
        "cinema_auto": "Tự động: {level}",
        "cinema_note": "Lý do sửa (được giữ trong báo cáo)",
        "cinema_reset": "Đưa so sánh này về tự động",
        "cinema_rules": "Cách xếp mức tự động",
        "cinema_rules_text": "- **Within-study bias** - nguy cơ sai lệch của từng nghiên cứu (Đặc điểm nghiên cứu), có trọng số theo mức đóng góp vào ước lượng (contribution matrix). Nghiên cứu chưa đánh giá được tính là *Some concerns*.\n- **Reporting bias** - kiểm định Egger trên funnel plot hiệu chỉnh so sánh: p < 0.10 → *Suspected*.\n- **Indirectness** - mức indirectness từng nghiên cứu (Đặc điểm nghiên cứu), tính như RoB; nghiên cứu chưa đánh giá được tính là *Low*.\n- **Imprecision** - CI so với khoảng tương đương: vượt một ngưỡng → *Some*, cả hai → *Major*.\n- **Heterogeneity** - khoảng dự đoán so với CI: PI vượt thêm một ngưỡng → *Some*, thêm hai → *Major*.\n- **Incoherence** - p của node-split (có cả trực tiếp và gián tiếp), nếu không thì kiểm định design-by-treatment: p ≤ 0.10 → *Some*, p ≤ 0.01 → *Major*.\n\nĐộ tin cậy bắt đầu ở **High**, hạ một mức cho mỗi *Some concerns* / *Suspected* và hai mức cho mỗi *Major concerns*. Các tiêu chí chồng lấn nhau (imprecision, heterogeneity, incoherence), nên hãy xem lại gợi ý và sửa kèm lý do khi cần (Nikolakopoulou và cs. 2020, PLoS Med).",
        "cinema_download": "Tải báo cáo CINeMA (CSV)",
        "col_comparison": "So sánh",
        "col_estimate": "Ước lượng (CI)",
        "col_pi": "Khoảng dự đoán",
        "col_direct": "% trực tiếp",
        "col_confidence": "Độ tin cậy",
        "dom_within_study_bias": "Sai lệch trong NC",
        "dom_reporting_bias": "Sai lệch báo cáo",
        "dom_indirectness": "Tính gián tiếp",
        "dom_imprecision": "Độ chính xác",
        "dom_heterogeneity": "Dị biệt",
        "dom_incoherence": "Không nhất quán",
        "lvl_no": "Không đáng lo",
        "lvl_some": "Có lo ngại",
        "lvl_major": "Lo ngại lớn",
        "lvl_undetected": "Không phát hiện",
        "lvl_suspected": "Nghi ngờ",
        "lvl_high": "Cao",
        "lvl_moderate": "Trung bình",
        "lvl_low": "Thấp",
        "lvl_very_low": "Rất thấp",
        "contrib_intro": "Tỷ lệ (%) đóng góp của từng so sánh trực tiếp (cột) vào từng ước lượng mạng lưới (dòng), tính từ hat matrix tách thành các luồng bằng chứng (Papakonstantinou và cs. 2018). Viền cam = bằng chứng trực tiếp của chính so sánh đó.",
        "contrib_estimate": "Ước lượng mạng lưới",
        "contrib_studies": "Đóng góp của từng nghiên cứu vào",
        "contrib_study": "Nghiên cứu",
        "contrib_share": "Đóng góp (%)",
        "contrib_rob": "Nguy cơ sai lệch",
        "contrib_download": "Tải contribution matrix (CSV)",
        "outcomes": "Outcome / phân tích",
        "outcomes_help": "Nhiều phân tích của cùng một mạng lưới trong một file (vd tử vong, nhập viện, tác dụng phụ). Treatments và đặc điểm nghiên cứu dùng chung; dữ liệu nghiên cứu, thiết lập mô hình và đánh giá CINeMA riêng cho từng phân tích.",
        "outcome_active": "Phân tích đang mở",
        "outcome_title": "Tên phân tích này",
        "outcome_new": "Phân tích mới",
        "outcome_new_name": "Tên",
        "outcome_copy": "Sao chép dữ liệu nghiên cứu từ phân tích hiện tại",
        "outcome_add": "Thêm",
        "outcome_delete": "Xoá phân tích này",
        "outcome_default": "Phân tích {n}",
    },
}

try:  # fr / es / zh / ko / ja
    from nma_editor_i18n import TRANSLATIONS as _TRANSLATIONS

    TEXT.update(_TRANSLATIONS)
except ImportError:  # pragma: no cover - English fallback
    pass

OUTCOMES = ["binary", "continuous", "rate", "contrast"]
CONTRAST_MEASURES = ["HR", "OR", "RR", "RATE_RATIO", "MD", "SMD", "RD"]
MEASURE_LABELS = {"OR": "OR (odds ratio)", "RR": "RR (risk ratio)", "RD": "RD (risk difference)",
                  "MD": "MD (mean difference)", "SMD": "SMD (Hedges' g)", "RATE_RATIO": "Rate ratio",
                  "HR": "HR (hazard ratio)"}


def _t(key: str) -> str:
    language = st.session_state.get("language", "en")
    return TEXT.get(language, TEXT["en"]).get(key) or TEXT["en"][key]


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def _load_document(document: dict[str, Any]) -> None:
    st.session_state.nma_document = document
    st.session_state.nma_base_treatments = sup.treatments_frame(document)
    st.session_state.nma_base_arms = sup.arms_frame(document)
    st.session_state.nma_base_contrasts = sup.contrasts_frame(document)
    st.session_state.nma_base_chars = sup.characteristics_frame(document)
    st.session_state.nma_editor_version = st.session_state.get("nma_editor_version", 0) + 1
    st.session_state.nma_result = None
    st.session_state.nma_result_hash = None
    st.session_state.nma_incons_result = None
    st.session_state.nma_incons_hash = None
    st.session_state.nma_bayes = None
    st.session_state.nma_bayes_hash = None


RESULT_KEYS = ("nma_result", "nma_result_hash", "nma_incons_result", "nma_incons_hash", "nma_bayes", "nma_bayes_hash")


def _nma_model_files() -> ModelFiles:
    """Save / Save as / Open-from-folder for the NMA file (the NMA editor
    keeps its own autosave and restore banner)."""

    def serialize() -> str | None:
        return json.dumps(sup.bundle_documents(_all_documents(), st.session_state.nma_active),
                          indent=2, ensure_ascii=False)

    def open_text(text: str) -> str | None:
        try:
            _load_file(*sup.documents_from_file(json.loads(text)))
        except (ValueError, KeyError, TypeError) as exc:
            return str(exc)
        return None

    return ModelFiles("nma", "NMA", "nma-model.decide", serialize, open_text)


def _load_file(documents: list[dict[str, Any]], active: int = 0) -> None:
    """Replace every analysis (open file / load example)."""

    st.session_state.nma_analyses = [sup.ensure_analysis_id(document) for document in documents]
    st.session_state.nma_active = active
    st.session_state.nma_analysis_cache = {}
    _load_document(st.session_state.nma_analyses[active])


def _stash_active() -> None:
    """Keep the active analysis (and its results) before leaving it."""

    document = st.session_state.nma_document
    st.session_state.nma_analyses[st.session_state.nma_active] = document
    st.session_state.nma_analysis_cache[document.get("analysis_id")] = {
        key: st.session_state.get(key) for key in RESULT_KEYS}


def _switch_analysis(index: int) -> None:
    _stash_active()
    target = sup.share_common(st.session_state.nma_document, st.session_state.nma_analyses[index])
    st.session_state.nma_active = index
    _load_document(target)
    for key, value in st.session_state.nma_analysis_cache.get(target.get("analysis_id"), {}).items():
        st.session_state[key] = value


def _all_documents() -> list[dict[str, Any]]:
    documents = list(st.session_state.nma_analyses)
    documents[st.session_state.nma_active] = st.session_state.nma_document
    return documents


def _init_state() -> None:
    if "nma_document" not in st.session_state:
        _load_file([sup.default_document()])
    if "nma_analyses" not in st.session_state:
        st.session_state.nma_analyses = [sup.ensure_analysis_id(st.session_state.nma_document)]
        st.session_state.nma_active = 0
        st.session_state.nma_analysis_cache = {}
    if "nma_base_chars" not in st.session_state:
        st.session_state.nma_base_chars = sup.characteristics_frame(st.session_state.nma_document)
    defaults = {
        "nma_results_open": False,
        "nma_results_collapsed": False,
        "nma_details_open": False,
        "nma_incons_open": False,
        "nma_incons_collapsed": False,
        "nma_incons_requested": False,
        "nma_ee_open": False,
        "nma_more_open": False,
        "nma_more_collapsed": False,
        "nma_ee_collapsed": False,
        "nma_details_collapsed": False,
        "nma_check_open": False,
        "nma_run_requested": False,
        "nma_layout_nonce": 0,
        "nma_view": "2D",
        "nma_color_mode": "class",
        "nma_size_by": "patients",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _current_treatment_names(version: int) -> list[str]:
    """Treatment names including edits not yet applied to the base table.

    The Treatments table renders below Model settings (layout order), so
    read its latest edit straight from the component's session value to
    keep the Reference dropdown current on the same rerun.
    """

    names = list(st.session_state.nma_base_treatments["name"].tolist())
    value = st.session_state.get(f"nma_treatments_table_{version}")
    if isinstance(value, dict) and value.get("version") == version:
        names = [row.get("name") for row in value.get("rows") or []]
    return [str(name).strip() for name in names if isinstance(name, str) and name.strip()]


def _commit_tables(treatments: pd.DataFrame, arms: pd.DataFrame, contrasts: pd.DataFrame,
                   chars: pd.DataFrame | None = None) -> None:
    """Make the current edits the new editor baseline (before a version bump)."""

    st.session_state.nma_base_treatments = treatments.reset_index(drop=True)
    st.session_state.nma_base_arms = arms.reset_index(drop=True)
    st.session_state.nma_base_contrasts = contrasts.reset_index(drop=True)
    if chars is not None:
        st.session_state.nma_base_chars = chars.reset_index(drop=True)
    st.session_state.nma_editor_version += 1


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CSS = """
<style>
div[data-testid="stExpander"] {
    background: #f7fbff;
    border: 1px solid #b7d0f4;
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stExpander"] details > summary { background: #2f6fd6; color: #ffffff; }
div[data-testid="stExpander"] details > summary:hover { background: #1d4f9b; color: #ffffff; }
div[data-testid="stExpander"] details > summary svg { color: #ffffff; }
div[class*="st-key-nma_toolbar_sticky"] {
    position: sticky;
    top: 0px;
    z-index: 999;
    margin-top: -24px;
    background: linear-gradient(135deg, #eaf3fc 0%, #dcebfa 100%);
    border-radius: 10px;
    padding: 6px 14px;
}
div.st-key-nma_check_network button { background: #dc2626; border-color: #b91c1c; color: #ffffff; }
div.st-key-nma_check_network button:hover { background: #b91c1c; border-color: #991b1b; color: #ffffff; }
div.st-key-nma_check_network button p { color: #ffffff; }
div.st-key-nma_run_model button { background: #16a34a; border-color: #15803d; color: #ffffff; }
div.st-key-nma_run_model button:hover { background: #15803d; border-color: #166534; color: #ffffff; }
div.st-key-nma_open_inconsistency button { background: #f97316; border-color: #ea580c; color: #ffffff; }
div.st-key-nma_open_inconsistency button:hover { background: #ea580c; border-color: #c2410c; color: #ffffff; }
div.st-key-nma_open_dashboard button { background: #1e3a8a; border-color: #1e293b; color: #ffffff; }
div.st-key-nma_open_dashboard button:hover { background: #17296b; border-color: #0f172a; color: #ffffff; }
div.st-key-nma_send_to_ee button { background: #2f6fd6; border-color: #1d4f9b; color: #ffffff; }
div.st-key-nma_send_to_ee button:hover { background: #1d4f9b; border-color: #1e3a8a; color: #ffffff; }
div.st-key-nma_open_more button { background: #7c3aed; border-color: #6d28d9; color: #ffffff; }
div.st-key-nma_open_more button:hover { background: #6d28d9; border-color: #5b21b6; color: #ffffff; }
div.st-key-nma_data_panel {
    background: #ffffff;
}
/* Network panel: dark-blue frame, white controls, white plot area. */
div.st-key-nma_network_panel {
    background: #1e3a8a;
    border-color: #1e3a8a !important;
    border-radius: 10px;
}
div.st-key-nma_network_panel h4,
div.st-key-nma_network_panel [data-testid="stWidgetLabel"] p,
div.st-key-nma_network_panel [data-testid="stMarkdownContainer"] p,
div.st-key-nma_network_panel [data-testid="stMarkdownContainer"] b {
    color: #ffffff !important;
}
div.st-key-nma_network_panel [data-testid="stMarkdownContainer"] span {
    color: #dbeafe !important;
}
div.st-key-nma_network_panel [data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #ffffff !important;
}
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #ffffff !important;
}
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button[kind$="Active"],
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button[data-testid$="Active"] {
    background: #bfdbfe !important;
    color: #1e3a8a !important;
    font-weight: 700;
}
/* Button labels are rendered through a markdown <p>; the white-text rule
   above for panel markdown would otherwise make them white on white. */
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button,
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button p,
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button span,
div.st-key-nma_network_panel [data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
}
div.st-key-nma_edge_info {
    background: #f3f7fe;
    border: 1px solid #bcd0f2;
    border-radius: 8px;
    padding: 10px 14px;
}
div.st-key-nma_network_panel div.st-key-nma_edge_info [data-testid="stMarkdownContainer"] p,
div.st-key-nma_network_panel div.st-key-nma_edge_info [data-testid="stMarkdownContainer"] b,
div.st-key-nma_network_panel div.st-key-nma_edge_info strong {
    color: #0f172a !important;
}
</style>
"""


# --------------------------------------------------------------------------
# Network plot (self-contained SVG; 3D is a JS projection, no extra packages)
# --------------------------------------------------------------------------

NETWORK_HTML = """
<div style="position:relative;font-family:'Source Sans Pro','Source Sans 3',system-ui,sans-serif">
<svg id="net" viewBox="0 0 600 400" width="100%" height="__HEIGHT__"
     style="background:#ffffff;border:1px solid #ffffff;border-radius:10px;touch-action:none;cursor:grab"
     role="img" aria-label="Network plot"></svg>
<div id="hint" style="position:absolute;left:12px;bottom:8px;font-size:12px;color:#475569"></div>
<button id="png" type="button" title="Download the network figure as a PNG image"
  style="position:absolute;top:8px;right:8px;display:flex;align-items:center;gap:6px;padding:6px 10px;
         font:600 13px 'Source Sans Pro','Source Sans 3',system-ui,sans-serif;color:#1e3a8a;background:#ffffff;
         border:1px solid #b7d0f4;border-radius:6px;cursor:pointer">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2.4"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M12 4v11"></path><path d="M7 10l5 5 5-5"></path><path d="M5 20h14"></path></svg>
  PNG
</button>
</div>
<script>
(function () {
  const DATA = __DATA__;
  const svg = document.getElementById('net');
  const NS = 'http://www.w3.org/2000/svg';
  const CX = 300, CY = 195;
  const is3d = DATA.view === '3D';
  let yaw = 0.6, pitch = 0.35, zoom = 1;
  const n = DATA.nodes.length;
  const pos = {};
  DATA.nodes.forEach(function (node, i) {
    if (is3d) {
      const k = n === 1 ? 0.5 : i / (n - 1);
      const y = 1 - 2 * k;
      const rad = Math.sqrt(Math.max(0, 1 - y * y));
      const th = i * 2.399963;
      pos[node.id] = {x: Math.cos(th) * rad * 140, y: y * 140, z: Math.sin(th) * rad * 140};
    } else {
      const a = -Math.PI / 2 + 2 * Math.PI * i / Math.max(n, 1);
      const R = n === 1 ? 0 : 140;
      pos[node.id] = {x: Math.cos(a) * R * 1.25, y: Math.sin(a) * R, z: 0};
    }
  });
  function project(p) {
    if (!is3d) return {x: CX + p.x * zoom, y: CY + p.y * zoom, z: 0, f: zoom};
    const x1 = p.x * Math.cos(yaw) + p.z * Math.sin(yaw);
    const z1 = -p.x * Math.sin(yaw) + p.z * Math.cos(yaw);
    const y2 = p.y * Math.cos(pitch) - z1 * Math.sin(pitch);
    const z2 = p.y * Math.sin(pitch) + z1 * Math.cos(pitch);
    const f = zoom * 520 / (520 + z2);
    return {x: CX + x1 * f, y: CY + y2 * f, z: z2, f: f};
  }
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const defs = el('defs', {});
    DATA.nodes.forEach(function (node, i) {
      const g = el('radialGradient', {id: 'g' + i, cx: '35%', cy: '30%', r: '72%'}, defs);
      el('stop', {offset: '0%', 'stop-color': '#ffffff'}, g);
      el('stop', {offset: '100%', 'stop-color': node.color}, g);
    });
    const P = {};
    DATA.nodes.forEach(function (node) { P[node.id] = project(pos[node.id]); });
    const zs = DATA.nodes.map(function (node) { return P[node.id].z; });
    const zmin = Math.min.apply(null, zs), zmax = Math.max.apply(null, zs);
    function depth(z) { return zmax > zmin ? (z - zmin) / (zmax - zmin) : 0; }
    const edges = DATA.edges.slice().sort(function (e1, e2) {
      return (P[e2.a].z + P[e2.b].z) - (P[e1.a].z + P[e1.b].z);
    });
    edges.forEach(function (e) {
      const a = P[e.a], b = P[e.b];
      const op = is3d ? 1 - 0.6 * depth((a.z + b.z) / 2) : 1;
      const line = el('line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y,
        stroke: e.selected ? '#1e3a8a' : '#7b8aa3', 'stroke-width': e.w * (a.f + b.f) / 2,
        'stroke-linecap': 'round', opacity: op});
      el('title', {}, line).textContent = e.a + ' vs ' + e.b + ': ' + e.n + (e.n === 1 ? ' study' : ' studies');
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      const badgeOp = is3d ? 0.55 + 0.45 * op : 1;
      el('rect', {x: mx - 11, y: my - 9, width: 22, height: 18, rx: 9,
        fill: e.selected ? '#1e3a8a' : '#ffffff', stroke: e.selected ? '#ffffff' : '#b7d0f4', 'stroke-width': 1.5, opacity: badgeOp});
      el('text', {x: mx, y: my + 4, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700,
        fill: e.selected ? '#ffffff' : '#334155', opacity: badgeOp}).textContent = e.n;
    });
    DATA.nodes.map(function (node, i) { return {node: node, i: i, p: P[node.id]}; })
      .sort(function (u, v) { return v.p.z - u.p.z; })
      .forEach(function (item) {
        const node = item.node, p = item.p, r = node.r * p.f;
        if (is3d) el('ellipse', {cx: p.x, cy: p.y + r + 4, rx: r * 0.9, ry: r * 0.25, fill: '#0f172a', opacity: 0.12});
        const c = el('circle', {cx: p.x, cy: p.y, r: r, fill: is3d ? 'url(#g' + item.i + ')' : node.color,
          stroke: '#ffffff', 'stroke-width': 2});
        el('title', {}, c).textContent = node.id + ': ' + node.studies + ' studies, ' + Math.round(node.patients) + ' patients';
        el('text', {x: p.x, y: p.y + r + 16, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600,
          fill: '#0f172a', stroke: '#ffffff', 'stroke-width': 3.5, 'paint-order': 'stroke'}).textContent = node.id;
      });
  }
  let drag = null;
  svg.addEventListener('pointerdown', function (ev) {
    drag = {x: ev.clientX, y: ev.clientY};
    svg.setPointerCapture(ev.pointerId);
    svg.style.cursor = 'grabbing';
  });
  svg.addEventListener('pointermove', function (ev) {
    if (!drag || !is3d) return;
    yaw += (ev.clientX - drag.x) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (ev.clientY - drag.y) * 0.01));
    drag = {x: ev.clientX, y: ev.clientY};
    draw();
  });
  function endDrag() { drag = null; svg.style.cursor = 'grab'; }
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);
  svg.addEventListener('wheel', function (ev) {
    ev.preventDefault();
    zoom = Math.max(0.5, Math.min(2.2, zoom * (ev.deltaY < 0 ? 1.1 : 0.9)));
    draw();
  }, {passive: false});
  /* PNG export: the figure exactly as currently shown (3D rotation and zoom
     included), on a white background, 3x resolution, with the colour legend
     (and a title) drawn underneath. */
  function exportPng() {
    const SCALE = 3, W = 600, H = 400;
    const legend = DATA.legend || [];
    const perRow = 3, rowH = 22;
    const legendH = legend.length ? 16 + Math.ceil(legend.length / perRow) * rowH : 0;
    const titleH = DATA.title ? 30 : 0;
    const totalH = titleH + H + legendH;
    const clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', NS);
    clone.setAttribute('width', W * SCALE);
    clone.setAttribute('height', totalH * SCALE);
    clone.setAttribute('viewBox', '0 ' + (-titleH) + ' ' + W + ' ' + totalH);
    clone.removeAttribute('style');
    const bg = document.createElementNS(NS, 'rect');
    bg.setAttribute('x', 0); bg.setAttribute('y', -titleH);
    bg.setAttribute('width', W); bg.setAttribute('height', totalH);
    bg.setAttribute('fill', '#ffffff');
    clone.insertBefore(bg, clone.firstChild);
    const font = "'Source Sans Pro','Source Sans 3',Arial,sans-serif";
    clone.setAttribute('font-family', font);
    if (DATA.title) {
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('x', W / 2); t.setAttribute('y', -titleH + 21);
      t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', 16);
      t.setAttribute('font-weight', 700); t.setAttribute('fill', '#0f172a');
      t.textContent = DATA.title;
      clone.appendChild(t);
    }
    legend.forEach(function (item, i) {
      const col = i % perRow, row = Math.floor(i / perRow);
      const x = 20 + col * (W - 40) / perRow, y = H + 16 + row * rowH;
      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', x + 6); dot.setAttribute('cy', y); dot.setAttribute('r', 6);
      dot.setAttribute('fill', item.color);
      clone.appendChild(dot);
      const label = document.createElementNS(NS, 'text');
      label.setAttribute('x', x + 18); label.setAttribute('y', y + 4.5);
      label.setAttribute('font-size', 12); label.setAttribute('fill', '#0f172a');
      label.textContent = item.label + (item.members ? ': ' + item.members : '');
      clone.appendChild(label);
    });
    clone.querySelectorAll('title').forEach(function (node) { node.remove(); });
    const text = new XMLSerializer().serializeToString(clone);
    const url = URL.createObjectURL(new Blob([text], {type: 'image/svg+xml;charset=utf-8'}));
    const img = new Image();
    img.onload = function () {
      const canvas = document.createElement('canvas');
      canvas.width = W * SCALE; canvas.height = totalH * SCALE;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob(function (blob) {
        if (!blob) return;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = DATA.filename || 'nma-network.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function () { URL.revokeObjectURL(link.href); }, 1500);
      }, 'image/png');
    };
    img.onerror = function () { URL.revokeObjectURL(url); };
    img.src = url;
  }
  document.getElementById('png').addEventListener('click', exportPng);
  window.__nmaExportPng = exportPng;
  document.getElementById('hint').textContent = is3d
    ? 'Drag to rotate · scroll to zoom · hover for details'
    : 'Scroll to zoom · hover for details';
  draw();
})();
</script>
<!-- nonce __NONCE__ -->
"""


def _render_network(payload: dict[str, Any], view: str, height: int, nonce: int) -> None:
    html = (NETWORK_HTML
            .replace("__DATA__", json.dumps({**payload, "view": view}).replace("</", "<\\/"))
            .replace("__HEIGHT__", str(height))
            .replace("__NONCE__", str(nonce)))
    st_html(html, height=height + 8)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def _forest_chart(fit, measure: str, reference: str, order: list[str]):
    ratio = is_ratio_measure(measure)
    rows = [{
        "treatment": effect.treatment,
        "estimate": effect.natural,
        "lower": effect.natural_lower,
        "upper": effect.natural_upper,
        "label": sup.format_effect(effect.natural, effect.natural_lower, effect.natural_upper),
        "pred_lower": effect.natural_pred_lower,
        "pred_upper": effect.natural_pred_upper,
    } for effect in fit.effects]
    frame = pd.DataFrame(rows)
    has_pi = frame["pred_lower"].notna().any()
    sort = [name for name in order if name in set(frame["treatment"])]
    scale = alt.Scale(type="log") if ratio else alt.Scale(zero=False)
    x_title = f"{measure} vs {reference}"
    base = alt.Chart(frame).encode(y=alt.Y("treatment:N", sort=sort, title=None,
                                          scale=alt.Scale(paddingInner=0.2, paddingOuter=0.5)))
    whiskers = base.mark_rule(strokeWidth=2.5, color="#1e3a8a").encode(
        x=alt.X("lower:Q", scale=scale, title=x_title), x2="upper:Q")
    points = base.mark_square(size=110, color="#1e3a8a").encode(
        x=alt.X("estimate:Q", scale=scale),
        tooltip=[alt.Tooltip("treatment:N"), alt.Tooltip("label:N", title=f"{measure} (CI)")])
    null = alt.Chart(pd.DataFrame({"x": [1.0 if ratio else 0.0]})).mark_rule(
        strokeDash=[4, 4], color="#334155").encode(x=alt.X("x:Q", scale=scale))
    layers = [null]
    if has_pi:
        layers.append(base.mark_rule(strokeWidth=7, color="#bfdbfe", opacity=0.9).encode(
            x=alt.X("pred_lower:Q", scale=scale, title=x_title), x2="pred_upper:Q"))
    # "OR (95% CI)" label centred above each line.
    labels = base.mark_text(dy=-14, fontSize=12, fontWeight=600, color="#0f172a").encode(
        x=alt.X("estimate:Q", scale=scale), text="label:N")
    layers += [whiskers, points, labels]
    return alt.layer(*layers).properties(height=max(140, 56 * len(frame) + 30))


def _pscore_chart(fit, order: list[str], label: str = "P-score"):
    frame = pd.DataFrame({"treatment": order, "score": [fit.p_scores[name] for name in order]})
    bars = alt.Chart(frame).mark_bar(color="#2f6fd6", cornerRadiusEnd=3).encode(
        x=alt.X("score:Q", title=label, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("treatment:N", sort=order, title=None),
        tooltip=["treatment", alt.Tooltip("score:Q", title=label, format=".2f")])
    labels = bars.mark_text(align="left", dx=4, color="#0f172a").encode(text=alt.Text("score:Q", format=".2f"))
    return (bars + labels).properties(height=max(120, 36 * len(frame) + 20))


def _uses_exact(spec: dict) -> bool:
    """Exact arm-level likelihood chosen and available for this outcome / measure."""

    available = (spec.get("outcome"), spec.get("measure")) in EXACT_LIKELIHOOD
    return available and spec.get("bayes", {}).get("likelihood", "exact") == "exact"


def _bayes_settings(spec: dict, effects: str, **override) -> BayesSettings:
    bayes = spec.get("bayes", {})
    values = dict(effects=effects, chains=int(bayes.get("chains", 4)), draws_per_chain=int(bayes.get("draws", 1000)),
                  seed=int(bayes.get("seed", 2026)), prior_sd=float(bayes.get("prior_sd", 100.0)),
                  tau_prior=bayes.get("tau_prior", "uniform"), tau_max=float(bayes.get("tau_max", 5.0)),
                  tau_scale=float(bayes.get("tau_scale", 0.5)), class_model=bayes.get("class_model", "none"),
                  class_sd_scale=float(bayes.get("class_sd_scale", 1.0)))
    if values["tau_prior"] == "lognormal":
        prior = _informative_tau(spec)
        if prior is None:
            values["tau_prior"] = "uniform"
        else:
            values.update(tau_mu=prior["mu"], tau_sigma=prior["sigma"], tau_df=prior["df"], tau_prior_label=prior["label"])
    values.update(override)
    return BayesSettings(**values)


def _informative_tau(spec: dict) -> dict | None:
    """The informative τ prior chosen in the sidebar, rescaled to the analysis
    scale (MD: shifted by the pooled SD of the data). None when unavailable."""

    scale, note = nma_priors.scale_for(spec.get("outcome", "binary"), spec.get("measure", "OR"))
    if scale is None:
        return None
    bayes = spec.get("bayes", {})
    entry = nma_priors.lookup(scale, bayes.get("tau_outcome", nma_priors.GENERAL),
                              bayes.get("tau_comparison", nma_priors.GENERAL))
    mu = entry.mu
    extra = ""
    if spec.get("measure") == "MD":
        sd = sup.pooled_sd(st.session_state.nma_document)
        if sd and sd > 0:
            mu = entry.mu + 2.0 * math.log(sd)
            extra = f" × pooled SD {sd:.3g}"
    cell = f"{entry.outcome}; {entry.comparison}" if entry.specific else _t("tau_general_used")
    label = f"{entry.source}: {cell}{extra}"
    return {"mu": mu, "sigma": entry.sigma, "df": entry.df, "label": label, "note": note, "entry": entry,
            "rescaled": nma_priors.TauPrior(scale, entry.outcome, entry.comparison, mu, entry.sigma, entry.df,
                                            entry.specific, entry.source)}


def _run_bayes(model, spec: dict) -> dict:
    """Normal approximation: both fixed- and random-effects posteriors (exact
    sampling, cheap), so switching the Effects model never needs a rerun.
    Exact likelihood (MCMC): only the selected effects model."""

    if _uses_exact(spec):
        warmup = int(spec.get("bayes", {}).get("warmup", 1000))
        runs = {}
        for effects in ("fixed", "random"):
            with st.spinner(_t("mcmc_running").format(effects=effects)):
                runs[effects] = run_nma_bayesian_exact(model, _bayes_settings(spec, effects), warmup=warmup)
        return runs
    return {effects: run_nma_bayesian(model, _bayes_settings(spec, effects)) for effects in ("fixed", "random")}


def _active_bayes():
    """The Bayesian result for the chosen effects model, when the Bayesian
    engine is selected and has been run; else None."""

    spec = st.session_state.nma_document.get("spec", {})
    runs = st.session_state.get("nma_bayes")
    if spec.get("engine") != "bayesian" or not runs:
        return None
    run = runs.get(spec.get("effects", "random"))
    return run if run is not None and not run.errors else None


def _selected_fit(result):
    bayes = _active_bayes()
    if bayes is not None:
        return bayes.fit
    effects = st.session_state.nma_document.get("spec", {}).get("effects", "random")
    return result.fixed if effects == "fixed" else result.random


def _rankogram_chart(rank_probabilities: dict, order: list[str]):
    frame = sup.rank_probability_frame(rank_probabilities, order)
    return alt.Chart(frame).mark_bar().encode(
        x=alt.X("rank:O", title="Rank (1 = best)"),
        y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("treatment:N", sort=order, legend=alt.Legend(orient="bottom", title=None)),
        xOffset=alt.XOffset("treatment:N", sort=order),
        tooltip=["treatment", "rank", alt.Tooltip("probability:Q", format=".2f")],
    ).properties(height=220)


def _render_results_body() -> None:
    result = st.session_state.get("nma_result")
    if result is None:
        st.info(_t("not_run"))
        return
    if st.session_state.get("nma_result_hash") != sup.document_hash(st.session_state.nma_document):
        st.warning(_t("stale"))
    fit = _selected_fit(result)
    het = result.heterogeneity
    bayes = _active_bayes()
    spec = st.session_state.nma_document.get("spec", {})
    if spec.get("engine") == "bayesian" and bayes is None:
        runs = st.session_state.get("nma_bayes") or {}
        failed = runs.get(spec.get("effects", "random"))
        if failed is not None and failed.errors:
            st.error("\n".join(getattr(e, "message", str(e)) for e in failed.errors))
        else:
            st.info(_t("bayes_not_run"))
    level = st.session_state.nma_document.get("spec", {}).get("ci_level", 0.95)
    if bayes is not None:
        if st.session_state.get("nma_bayes_hash") != sup.bayes_hash(st.session_state.nma_document):
            st.warning(_t("stale"))
        chips = [
            f"{_t('bayesian')} · {_t('chip_random') if bayes.tau is not None else _t('chip_fixed')}",
            (_t("lik_chip_exact").format(kind=LIKELIHOOD_LABELS.get(bayes.likelihood, bayes.likelihood)) if isinstance(bayes, ExactResult)
             else _t("lik_chip_normal")),
            f"{result.measure} · {_t('chip_reference')}: {result.reference}",
            (f"τ {bayes.tau.median:.3f} ({bayes.tau.lower:.3f}–{bayes.tau.upper:.3f})" if bayes.tau else "τ = 0"),
            *([_t("chip_tau_prior").format(label=bayes.settings.tau_prior_label or "informative")]
              if bayes.tau is not None and getattr(bayes.settings, "tau_prior", "") == "lognormal" else []),
            _t("chip_dres").format(dres=bayes.residual_deviance, n=bayes.n_data_points),
            f"DIC {bayes.dic:.1f}",
            f"{level:.0%} CrI",
        ]
        # The method note lives in the (?) tooltip instead of a line of text.
        st.markdown(" &nbsp;·&nbsp; ".join(f"`{chip}`" for chip in chips),
                    help=_t("bayes_note_results").format(pct=f"{level:.0%}"))
        if sup.is_example_data(st.session_state.nma_document):
            st.caption(_t("illustrative"))
        order = sup.ranking_order(fit)
        forest_col, rank_col = st.columns([1.3, 1], gap="medium")
        with forest_col:
            st.markdown(f"**{_t('forest')}**")
            st.altair_chart(_forest_chart(fit, result.measure, result.reference, order), use_container_width=True)
        with rank_col:
            st.markdown(f"**{_t('sucra')}**")
            st.altair_chart(_pscore_chart(fit, order, "SUCRA"), use_container_width=True)
            _show_table(pd.DataFrame([{
                _t("sucra_col_treatment"): name,
                "SUCRA": round(bayes.sucra[name], 3),
                _t("sucra_col_mean_rank"): round(bayes.mean_rank[name], 2),
                _t("sucra_col_best"): round(bayes.rank_probabilities[name][0], 3),
            } for name in order]))
        st.markdown(f"**{_t('rankogram')}**")
        st.altair_chart(_rankogram_chart(bayes.rank_probabilities, order), use_container_width=True)
        st.markdown(f"**{_t('league')} · {result.measure}**")
        st.caption(_t("league_hint"))
        league = sup.league_frame(fit, order)
        _show_table(league, index=True)
        st.download_button(_t("download_csv"), league.to_csv().encode("utf-8"), "nma-league-table-bayesian.csv",
                           "text/csv", key="nma_league_download_bayes")
        _render_absolute(fit, result.measure)
        return
    chips = [
        _t('chip_random') if fit.model == 'random' else _t('chip_fixed'),
        f"{result.measure} · {_t('chip_reference')}: {result.reference}",
        f"τ² = {het.tau2:.3f}" if fit.model == "random" else "τ² = 0",
        f"Q = {het.q:.2f}, df = {het.df}" + (f", p = {het.p_value:.3f}" if het.p_value is not None else ""),
        f"I² = {het.i2 * 100:.0f}%" if het.i2 is not None else "I² –",
    ]
    st.markdown(" &nbsp;·&nbsp; ".join(f"`{chip}`" for chip in chips))
    if sup.is_example_data(st.session_state.nma_document):
        st.caption(_t("illustrative"))

    order = sup.ranking_order(fit)
    forest_col, rank_col = st.columns([1.3, 1], gap="medium")
    with forest_col:
        st.markdown(f"**{_t('forest')}**")
        st.altair_chart(_forest_chart(fit, result.measure, result.reference, order), use_container_width=True)
        if fit.model == "random" and result.heterogeneity.prediction_df:
            st.caption(_t("prediction_note").format(
                pct=f"{st.session_state.nma_document.get('spec', {}).get('ci_level', 0.95):.0%}",
                df=result.heterogeneity.prediction_df))
    with rank_col:
        st.markdown(f"**{_t('rankings')}**")
        st.altair_chart(_pscore_chart(fit, order), use_container_width=True)

    st.markdown(f"**{_t('league')} · {result.measure}**")
    st.caption(_t("league_hint"))
    league = sup.league_frame(fit, order)
    _show_table(league, index=True)
    st.download_button(
        _t("download_csv"), league.to_csv().encode("utf-8"), "nma-league-table.csv", "text/csv",
        key="nma_league_download",
    )
    _render_absolute(fit, result.measure)


def _render_convergence(bayes) -> None:
    """Trace, density and running-mean plots per chain for one parameter."""

    if getattr(bayes, "chain_draws", None) is None:
        return
    st.markdown(f"**{_t('conv_title')}**")
    st.caption(_t("conv_help"))
    names = list(bayes.parameter_names) + (["tau"] if getattr(bayes, "chain_tau", None) is not None else [])
    diag = {d.name: d for d in bayes.diagnostics}
    worst = max(names, key=lambda n: diag[n].rhat if n in diag else 0.0)
    pick_col, info_col = st.columns([1.2, 2])
    with pick_col:
        name = st.selectbox(_t("conv_parameter"), names, index=names.index(worst),
                            format_func=lambda n: f"{n}  (R-hat {diag[n].rhat:.3f}, ESS {diag[n].ess:.0f})" if n in diag else n,
                            key=f"nma_conv_param_{id(bayes)}")
    values = bayes.chain_tau if name == "tau" else bayes.chain_draws[:, :, names.index(name)]
    n_chain, n_draw = values.shape
    step = max(1, n_draw // 600)
    rows = [{"chain": f"{_t('conv_chain')} {c + 1}", "iteration": i, "value": float(values[c, i])}
            for c in range(n_chain) for i in range(0, n_draw, step)]
    frame = pd.DataFrame(rows)
    running = pd.DataFrame([{"chain": f"{_t('conv_chain')} {c + 1}", "iteration": i,
                             "value": float(np.mean(values[c, :i + 1]))}
                            for c in range(n_chain) for i in range(0, n_draw, step)])
    with info_col:
        if name in diag:
            st.caption(_t("conv_summary").format(median=diag[name].median, lower=diag[name].lower, upper=diag[name].upper,
                                                 rhat=diag[name].rhat, ess=diag[name].ess))
    color = alt.Color("chain:N", legend=alt.Legend(orient="bottom", title=None))
    trace = alt.Chart(frame).mark_line(strokeWidth=0.8, opacity=0.8).encode(
        x=alt.X("iteration:Q", title=_t("conv_iteration")), y=alt.Y("value:Q", title=name, scale=alt.Scale(zero=False)),
        color=color).properties(height=220, title=_t("conv_trace"))
    density = alt.Chart(frame).transform_density("value", groupby=["chain"], as_=["value", "density"]).mark_area(
        opacity=0.35).encode(x=alt.X("value:Q", title=name), y=alt.Y("density:Q", title=None), color=color
    ).properties(height=220, title=_t("conv_density"))
    mean_chart = alt.Chart(running).mark_line(strokeWidth=1.2).encode(
        x=alt.X("iteration:Q", title=_t("conv_iteration")), y=alt.Y("value:Q", title=_t("conv_running"), scale=alt.Scale(zero=False)),
        color=color).properties(height=220, title=_t("conv_running"))
    left, right = st.columns([1.6, 1])
    with left:
        st.altair_chart(trace, use_container_width=True)
    with right:
        st.altair_chart(density, use_container_width=True)
    st.altair_chart(mean_chart, use_container_width=True)
    bad = [n for n in names if n in diag and (diag[n].rhat > 1.01 or diag[n].ess < 400)]
    if bad:
        st.warning(_t("conv_warning").format(params=", ".join(bad)))


def _render_class_effects(bayes) -> None:
    """Class effects of a Bayesian run with treatment classes: table, sigma_class and the DIC comparison
    with the model without classes (fitted on request)."""

    requested = getattr(bayes.settings, "class_model", "none")
    if requested == "none":
        return
    if bayes.class_model == "none" or not bayes.class_effects:
        notes = [w for w in bayes.warnings if "class" in w.lower()]
        st.caption(_t("class_not_used") + " " + " ".join(notes))
        return
    document = st.session_state.nma_document
    st.markdown(f"**{_t('class_title')}**")
    st.caption(_t("class_common_help" if bayes.class_model == "common" else "class_exchangeable_help"))
    _show_table(sup.class_effects_frame(bayes, bayes.measure, "CrI"))
    if bayes.class_sd is not None:
        sd = bayes.class_sd
        st.caption(_t("class_sd_text").format(median=sd.median, lower=sd.lower, upper=sd.upper))
    _show_warnings([w for w in bayes.warnings if "class" in w.lower()])
    key = (sup.bayes_hash(document), bayes.settings.effects)
    if st.button(_t("class_compare_button"), key="nma_class_compare_button"):
        with st.spinner(_t("mcmc_running").format(effects=bayes.settings.effects)):
            plain = run_nma_bayesian_exact(resolve_nma(document), dc_replace(bayes.settings, class_model="none"),
                                           warmup=bayes.warmup)
        st.session_state.nma_class_compare = {"key": key, "run": plain}
    stored = st.session_state.get("nma_class_compare")
    if stored and stored["key"] == key and not stored["run"].errors:
        plain = stored["run"]
        st.caption(_t("class_compare_text").format(dic_c=bayes.dic, dic_n=plain.dic, diff=bayes.dic - plain.dic,
                                                   pd_c=bayes.p_d, pd_n=plain.p_d))


def _render_model_fit(bayes) -> None:
    """TSD 2 model fit: fixed vs random DIC, residual deviance and the leverage plot."""

    st.markdown(f"**{_t('fit_title')}**")
    st.caption(_t("fit_hint").format(dres=bayes.residual_deviance, n=bayes.n_data_points, pd=bayes.p_d, dic=bayes.dic))
    runs = st.session_state.get("nma_bayes") or {}
    both = [(effects, runs.get(effects)) for effects in ("fixed", "random")
            if isinstance(runs.get(effects), ExactResult) and not runs[effects].errors]
    if len(both) == 2:
        _show_table(pd.DataFrame([{
            _t("ume_model"): _t(effects), "Dres": round(run.residual_deviance, 2), "pD": round(run.p_d, 2),
            "DIC": round(run.dic, 2),
            "τ (CrI)": f"{run.tau.median:.3f} ({run.tau.lower:.3f}–{run.tau.upper:.3f})" if run.tau else "–",
        } for effects, run in both]))
        gap = both[0][1].dic - both[1][1].dic
        st.caption(_t("fit_fe_re").format(better=_t("random") if gap > 3 else _t("fixed"), gap=abs(gap)))
    if bayes.acceptance:
        st.caption(_t("fit_acceptance") + ": " + ", ".join(f"{name} {value:.0%}" for name, value in bayes.acceptance.items()))
    points = pd.DataFrame([{
        "study": point.study_id, "data": point.label,
        "x": point.sign * math.sqrt(max(point.dev_bar, 0.0)), "leverage": point.leverage, "dev": point.dev_bar,
        "outside": (point.sign * math.sqrt(max(point.dev_bar, 0.0))) ** 2 + point.leverage > 3,
    } for point in bayes.data_points])
    if not points.empty:
        span = max(3.0, float(points["x"].abs().max()) * 1.1)
        curves = pd.DataFrame([{"x": x, "y": c - x * x, "c": f"x² + leverage = {c}"}
                               for c in (1, 2, 3) for x in np.linspace(-math.sqrt(c), math.sqrt(c), 41)])
        parabola = alt.Chart(curves).mark_line(strokeDash=[4, 4], color="#94a3b8").encode(
            x=alt.X("x:Q", title=_t("fit_x"), scale=alt.Scale(domain=[-span, span])),
            y=alt.Y("y:Q", title=_t("fit_y")), detail="c:N")
        dots = alt.Chart(points).mark_circle(size=80, opacity=0.85).encode(
            x="x:Q", y="leverage:Q",
            color=alt.condition("datum.outside", alt.value("#dc2626"), alt.value("#1e3a8a")),
            tooltip=["study", "data", alt.Tooltip("dev:Q", title="Dres", format=".2f"),
                     alt.Tooltip("leverage:Q", format=".2f")])
        st.altair_chart(alt.layer(parabola, dots).properties(height=300), use_container_width=True)
        st.caption(_t("fit_leverage_note"))
        flagged = points[points["outside"]]
        if not flagged.empty:
            st.warning(_t("fit_outliers").format(points=", ".join(f"{row.study} ({row.data})" for row in flagged.itertuples())))



def _render_ume(bayes) -> None:
    """TSD 4: consistency vs unrelated-mean-effects model (Bayesian, exact likelihood)."""

    st.markdown(f"**{_t('ume_title')}**")
    st.caption(_t("ume_help"))
    document = st.session_state.nma_document
    spec = document.get("spec", {})
    key = (sup.bayes_hash(document), spec.get("effects", "random"))
    stored = st.session_state.get("nma_ume")
    if st.button(_t("ume_run"), icon=":material/compare_arrows:", key="nma_ume_run"):
        try:
            model = resolve_nma(document)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner(_t("ume_run")):
            ume = run_nma_bayesian_exact(model, _bayes_settings(spec, spec.get("effects", "random")),
                                         model_type="ume", warmup=int(spec.get("bayes", {}).get("warmup", 1000)))
        stored = {"key": key, "ume": ume}
        st.session_state.nma_ume = stored
    if not stored:
        return
    if stored["key"] != key:
        st.warning(_t("stale"))
    ume = stored["ume"]
    if ume.errors:
        st.error("; ".join(error.message for error in ume.errors))
        return
    rows = []
    for label, run in ((_t("ume_consistency"), bayes), ("UME", ume)):
        rows.append({_t("ume_model"): label, "Dres": round(run.residual_deviance, 2), "pD": round(run.p_d, 2),
                     "DIC": round(run.dic, 2), _t("ume_points"): run.n_data_points,
                     "τ (CrI)": f"{run.tau.median:.3f} ({run.tau.lower:.3f}–{run.tau.upper:.3f})" if run.tau else "–"})
    _show_table(pd.DataFrame(rows))
    difference = bayes.dic - ume.dic
    (st.warning if difference > 3 else st.success)(
        _t("ume_worse" if difference > 3 else "ume_ok").format(diff=difference))
    by_point = {(p.study_id, p.label): p.dev_bar for p in ume.data_points}
    pairs = pd.DataFrame([{"study": p.study_id, "data": p.label, "consistency": p.dev_bar,
                           "ume": by_point.get((p.study_id, p.label))} for p in bayes.data_points]).dropna()
    if not pairs.empty:
        top = float(max(pairs["consistency"].max(), pairs["ume"].max(), 1.0)) * 1.05
        line = alt.Chart(pd.DataFrame({"x": [0.0, top], "y": [0.0, top]})).mark_line(
            strokeDash=[4, 4], color="#94a3b8").encode(x="x:Q", y="y:Q")
        dots = alt.Chart(pairs).mark_circle(size=80, color="#7c3aed", opacity=0.85).encode(
            x=alt.X("consistency:Q", title=_t("ume_x"), scale=alt.Scale(domain=[0, top])),
            y=alt.Y("ume:Q", title=_t("ume_y"), scale=alt.Scale(domain=[0, top])),
            tooltip=["study", "data", alt.Tooltip("consistency:Q", format=".2f"), alt.Tooltip("ume:Q", format=".2f")])
        st.altair_chart(alt.layer(line, dots).properties(height=300), use_container_width=True)
        st.caption(_t("ume_devdev_note"))
    network = bayes.fit.league if bayes.fit else {}
    _show_table(pd.DataFrame([{
        _t("col_comparison"): f"{arm} vs {base}",
        f"UME {bayes.measure}": sup.format_effect(effect.natural, effect.natural_lower, effect.natural_upper),
        f"NMA {bayes.measure}": (sup.format_effect(network[(arm, base)].natural, network[(arm, base)].natural_lower,
                                                   network[(arm, base)].natural_upper) if (arm, base) in network else "–"),
    } for (arm, base), effect in ume.ume_effects.items()]))


def _render_absolute(fit, measure: str) -> None:
    """Absolute risks per 1000 and NNT for a chosen risk with the reference."""

    if measure not in ABSOLUTE_MEASURES or not fit.effects:
        return
    document = st.session_state.nma_document
    settings = document.setdefault("absolute", {})
    try:
        observed = observed_baseline_risk(resolve_nma(document))
    except ValueError:
        observed = None
    st.markdown(f"**{_t('abs_title')}**")
    input_col, note_col = st.columns([1, 2], vertical_alignment="bottom")
    with input_col:
        default = settings.get("baseline_risk") or observed or 0.10
        percent = st.number_input(_t("abs_baseline").format(ref=fit.effects[0].versus), min_value=0.01,
                                  max_value=99.99, step=0.5, format="%.2f", value=float(round(default * 100, 2)),
                                  key=f"nma_abs_p0_{document.get('analysis_id', '')}")
        settings["baseline_risk"] = percent / 100.0
    with note_col:
        hint = _t("abs_hint")
        if observed is not None:
            hint += " " + _t("abs_observed").format(pct=f"{observed * 100:.1f}%")
        if measure in ("HR", "RATE_RATIO"):
            hint += " " + _t("abs_hr_note")
        st.caption(hint)
    small = document.get("spec", {}).get("small_values", "good")
    rows = absolute_effects(fit, measure, percent / 100.0, small)
    per = 1000.0
    frame = pd.DataFrame([{
        _t("abs_col_treatment"): f"{row.treatment} vs {row.versus}",
        _t("abs_col_ref"): f"{row.baseline_risk * per:.0f}",
        _t("abs_col_risk"): f"{row.risk * per:.0f} ({row.risk_lower * per:.0f} to {row.risk_upper * per:.0f})",
        _t("abs_col_diff"): (f"{row.difference * per:+.0f} ({row.difference_lower * per:+.0f} to "
                             f"{row.difference_upper * per:+.0f})"),
        "NNT": row.nnt_text,
    } for row in rows])
    _show_table(frame)
    st.caption(_t("abs_nnt_note"))
    st.download_button(_t("download_csv"), frame.to_csv(index=False).encode("utf-8"), "nma-absolute-effects.csv",
                       "text/csv", key=f"nma_abs_download_{measure}_{fit.model}")


def _render_details_body() -> None:
    result = st.session_state.get("nma_result")
    if result is None:
        st.info(_t("not_run"))
        return
    bayes = _active_bayes()
    if bayes is not None:
        st.markdown(f"**{_t('bayes_diag')}**")
        _show_table(pd.DataFrame([{
            "Parameter": diag.name, "Mean": round(diag.mean, 4), "SD": round(diag.sd, 4),
            "Median": round(diag.median, 4), "CrI lower": round(diag.lower, 4), "CrI upper": round(diag.upper, 4),
            "R-hat": round(diag.rhat, 3), "ESS": int(round(diag.ess)),
        } for diag in bayes.diagnostics]), flags=[diag.rhat > 1.01 for diag in bayes.diagnostics])
        st.markdown(f"**{_t('bayes_fit')}**")
        _show_table(pd.DataFrame([{
            "Residual deviance": round(bayes.residual_deviance, 2), "Data points": bayes.n_data_points,
            "pD": round(bayes.p_d, 2), "DIC": round(bayes.dic, 2),
            "Draws": bayes.settings.chains * bayes.settings.draws_per_chain,
        }]))
    if isinstance(bayes, ExactResult):
        _render_convergence(bayes)
        _render_model_fit(bayes)
        _render_class_effects(bayes)
    het = result.heterogeneity
    st.markdown(f"**{_t('heterogeneity')}**")
    _show_table(pd.DataFrame([{
        "Q": round(het.q, 3), "df": het.df,
        "p": None if het.p_value is None else round(het.p_value, 4),
        "I²": None if het.i2 is None else f"{het.i2 * 100:.1f}%",
        "τ²": round(het.tau2, 4), "τ": round(het.tau, 4),
    }]))

    rows = []
    for fit in (result.fixed, result.random):
        for effect in fit.effects:
            rows.append({
                "model": fit.model, "treatment": effect.treatment, "vs": effect.versus,
                result.measure: round(effect.natural, 4),
                "lower": round(effect.natural_lower, 4), "upper": round(effect.natural_upper, 4),
                "estimate (analysis scale)": round(effect.estimate, 5), "SE": round(effect.se, 5),
                "P-score": round(fit.p_scores[effect.treatment], 3),
                "PI lower": None if effect.natural_pred_lower is None else round(effect.natural_pred_lower, 4),
                "PI upper": None if effect.natural_pred_upper is None else round(effect.natural_pred_upper, 4),
            })
    _show_table(pd.DataFrame(rows))

    st.markdown(f"**{_t('observations')}**")
    _show_table(pd.DataFrame([{
        "study": obs.study_id, "treatment": obs.treatment, "vs": obs.versus,
        "estimate": round(obs.estimate, 5), "variance": round(obs.variance, 6),
    } for obs in result.observations]), max_height=360)
    if result.excluded_studies:
        st.markdown(f"**{_t('excluded')}:** {', '.join(result.excluded_studies)}")
    _show_warnings(result.warnings)
    _render_rob_sensitivity(result)
    _render_leave_one_out(result)
    _render_prior_sensitivity()


COLUMN_KEYS = {
    "Comparison": "col_comparison", "Direct studies": "incons_col_direct_n", "Study": "contrib_study",
    "Studies": "studies", "Treatment": "rank_treatment", "Parameter": "colh_parameter", "Mean": "conv_mean",
    "Median": "conv_median_v", "Estimate": "conv_estimate", "CI lower": "colh_ci_lower", "CI upper": "colh_ci_upper",
    "CrI lower": "colh_cri_lower", "CrI upper": "colh_cri_upper", "model": "colh_model", "treatment": "rank_treatment",
    "vs": "colh_vs", "lower": "conv_lower", "upper": "conv_upper", "estimate (analysis scale)": "colh_estimate_scale",
    "PI lower": "colh_pi_lower", "PI upper": "colh_pi_upper", "study": "contrib_study", "estimate": "conv_estimate",
    "variance": "colh_variance", "Study removed": "colh_study_removed", "Conclusion changes for": "colh_conclusion",
    "design": "colh_design", "studies": "studies", "comparison": "col_comparison", "patients": "patients",
    "study ids": "colh_study_ids", "Year": "colh_year", "Mean age": "colh_mean_age", "Female %": "colh_female",
    "Follow-up (months)": "colh_follow_up", "Risk of bias": "contrib_rob", "Data points": "ume_points",
    "Model": "ume_model", "Network estimate": "contrib_estimate", "Direct evidence %": "col_direct",
    "Confidence": "col_confidence", "Ratio (indirect / direct)": "colh_ratio_id",
    "Difference (direct - indirect)": "incons_col_difference", "Draws": "draws", "β for": "colh_beta_for",
    "Prior for τ": "prior_col",
}
COLUMN_PREFIXES = [("Direct ", "direct"), ("Indirect ", "incons_col_indirect"), ("Network ", "colh_network"),
                   ("All studies ", "colh_all_studies"), ("Without high RoB ", "colh_without_rob"),
                   ("Base (", "colh_base"), ("UME ", "colh_ume"), ("NMA ", "colh_nma")]


def _column_label(name) -> str:
    """Translated column heading for the read-only tables (English otherwise)."""

    text = str(name)
    key = COLUMN_KEYS.get(text)
    if key and key in TEXT["en"]:
        return _t(key)
    for prefix, prefix_key in COLUMN_PREFIXES:
        if text.startswith(prefix) and prefix_key in TEXT["en"]:
            rest = text[len(prefix):]
            return f"{_t(prefix_key)} ({rest}" if prefix == "Base (" else f"{_t(prefix_key)} {rest}"
    return text


def _show_table(frame: pd.DataFrame, flags=None, index: bool = False, max_height: int | None = None) -> None:
    frame = frame.rename(columns=_column_label)
    if index and frame.index.name:
        frame.index.name = _column_label(frame.index.name)
    st.markdown(sup.html_table(frame, flags=flags, index=index, max_height=max_height), unsafe_allow_html=True)


WARNING_PATTERNS = [
    (r"^(?P<id>.+?): zero cell - (?P<x>[0-9.]+) added to every cell of this study$", "warn_zero_cell"),
    (r"^(?P<id>.+?): excluded - every arm has 0 events.*$", "warn_excluded_zero"),
    (r"^(?P<id>.+?): excluded - no events \(or all events\) in every arm.*$", "warn_excluded_zero"),
    (r"^(?P<id>.+?): excluded - an arm has zero variance.*$", "warn_zero_variance"),
    (r"^No degrees of freedom for heterogeneity.*$", "warn_no_df"),
    (r"^Only (?P<n>\d+) points: the funnel plot.*$", "warn_funnel_points"),
    (r"^No comparison has both direct and independent indirect evidence.*$", "incons_none"),
    (r"^No residual degrees of freedom.*$", "warn_no_residual_df"),
    (r"^No risk-of-bias judgements entered.*$", "warn_no_rob"),
    (r"^Low acceptance in (?P<which>.+?) - check.*$", "warn_low_acceptance"),
]


def _warning_text(text: str) -> str:
    import re

    for pattern, key in WARNING_PATTERNS:
        match = re.match(pattern, text)
        if match:
            return _t(key).format(**match.groupdict())
    return text


def _show_warnings(warnings) -> None:
    for warning in warnings:
        st.caption(f"• {_warning_text(str(warning))}")


def _render_characteristic_variables(document: dict, treatments_df, arms_df, contrasts_df, chars_df) -> None:
    """Expander under the Study characteristics table: rename, retype, add or
    remove the variable columns (stored in document["characteristic_columns"])."""

    variables = sup.characteristic_variables(document)
    v = st.session_state.nma_editor_version  # widget keys follow the editor version: a reset must not replay old edits
    type_labels = {"number": _t("chars_type_number"), "text": _t("chars_type_text")}

    def apply(new_variables: list[dict[str, str]]) -> None:
        document["characteristic_columns"] = new_variables
        refreshed = sup.characteristics_frame({
            "studies": sup.studies_from_frames(arms_df, contrasts_df),
            "study_characteristics": sup.characteristics_from_frame(chars_df),
            "characteristic_columns": new_variables,
        })
        _commit_tables(treatments_df, arms_df, contrasts_df, refreshed)
        st.rerun()

    with st.expander(_t("chars_vars_title"), icon=":material/tune:", expanded=False):
        st.caption(_t("chars_vars_help"))
        changed = False
        updated: list[dict[str, str]] = []
        for index, variable in enumerate(variables):
            cols = st.columns([3, 1.6, 0.6], vertical_alignment="bottom")
            with cols[0]:
                label = st.text_input(_t("chars_var_label"), value=variable["label"],
                                      key=f"nma_chars_var_label_{v}_{variable['key']}",
                                      label_visibility="visible" if index == 0 else "collapsed")
            with cols[1]:
                kind = st.selectbox(_t("chars_var_type"), ["number", "text"],
                                    index=0 if variable["type"] == "number" else 1,
                                    format_func=lambda k: type_labels[k], key=f"nma_chars_var_type_{v}_{variable['key']}",
                                    label_visibility="visible" if index == 0 else "collapsed")
            with cols[2]:
                if index == 0:
                    st.caption(" ")
                remove = st.button("", icon=":material/delete:", key=f"nma_chars_var_del_{v}_{variable['key']}",
                                   help=_t("chars_var_delete").format(name=variable["label"]))
            if remove:
                changed = True
                continue
            label = (label or "").strip() or variable["label"]
            if label != variable["label"] or kind != variable["type"]:
                changed = True
            updated.append({"key": variable["key"], "label": label, "type": kind})
        add_cols = st.columns([3, 1.6, 0.6], vertical_alignment="bottom")
        with add_cols[0]:
            new_label = st.text_input(_t("chars_var_new"), value="", key=f"nma_chars_var_new_{v}",
                                      placeholder="e.g. Mean BMI, Region, Dose")
        with add_cols[1]:
            new_type = st.selectbox(_t("chars_var_type"), ["number", "text"], format_func=lambda k: type_labels[k],
                                    key=f"nma_chars_var_newtype_{v}", label_visibility="collapsed")
        with add_cols[2]:
            add = st.button("", icon=":material/add:", key=f"nma_chars_var_add_{v}",
                            help=_t("chars_var_add"))
        if add and (new_label or "").strip():
            key = sup.variable_key(new_label.strip(), [v["key"] for v in updated])
            updated.append({"key": key, "label": new_label.strip(), "type": new_type})
            changed = True
        if changed:
            apply(updated)
        if document.get("characteristic_columns") is not None and st.button(
                _t("chars_var_reset"), icon=":material/restart_alt:", key=f"nma_chars_var_reset_{v}"):
            document.pop("characteristic_columns", None)
            refreshed = sup.characteristics_frame({
                "studies": sup.studies_from_frames(arms_df, contrasts_df),
                "study_characteristics": sup.characteristics_from_frame(chars_df),
            })
            _commit_tables(treatments_df, arms_df, contrasts_df, refreshed)
            st.rerun()


def _table_editor(key: str, base: pd.DataFrame, column_spec: list[dict], all_columns: list[str],
                  numeric: list[str], version: int, max_height: int = 420) -> pd.DataFrame:
    """Editable table with a dark-blue header (nma_table_component).

    Returns the edited table for this editor version, or the baseline when
    nothing has been edited yet. `base` keeps every column (hidden ones
    included), so switching outcome type never drops data."""

    rows = nma_table(
        columns=column_spec, rows=sup.frame_to_rows(base), version=version, max_height=max_height,
        add_label=_t("add_row"), delete_label=_t("delete_row"),
        undo_label=_t("undo"), redo_label=_t("redo"),
        paste_labels={name: _t(f"paste_{name}") for name in ("label", "hint", "replace", "append", "close", "empty")},
        key=f"{key}_{version}",
    )
    if rows is None:
        return base
    return sup.rows_to_frame(rows, all_columns, numeric)


def _fmt_p(value) -> str:
    if value is None:
        return "–"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _render_inconsistency_body() -> None:
    spec = st.session_state.nma_document.setdefault("spec", {})
    method_labels = {"side": _t("split_side"), "sidde": _t("split_sidde")}
    chosen = st.segmented_control(_t("split_method"), ["side", "sidde"], default=_split_method(),
                                  format_func=lambda key: method_labels[key], help=_t("split_method_help"),
                                  key="nma_split_method_control") or "side"
    if chosen != _split_method():
        spec["split_method"] = chosen
        st.session_state.nma_incons_requested = True
        st.rerun()
    incons = st.session_state.get("nma_incons_result")
    if incons is None:
        st.info(_t("not_run"))
        return
    if st.session_state.get("nma_incons_hash") != _incons_hash():
        st.warning(_t("stale"))
    if incons.errors:
        st.error("\n".join(error.message for error in incons.errors))
        return

    dbt = incons.design_by_treatment
    st.markdown(f"**{_t('incons_global')}**")
    rows = [
        {"": _t("incons_row_total"), "Q": dbt.q_total, "df": dbt.df_total, "p": None},
        {"": _t("incons_row_within"), "Q": dbt.q_within, "df": dbt.df_within, "p": dbt.p_within},
        {"": _t("incons_row_between"), "Q": dbt.q_between, "df": dbt.df_between, "p": dbt.p_between},
    ]
    if dbt.q_between_random is not None:
        rows.append({"": _t("incons_row_between_re"), "Q": dbt.q_between_random, "df": dbt.df_between,
                     "p": dbt.p_between_random})
    _show_table(pd.DataFrame([{**row, "Q": round(row["Q"], 3), "p": _fmt_p(row["p"])} for row in rows]))
    check_p = dbt.p_between_random if dbt.p_between_random is not None else dbt.p_between
    if check_p is not None:
        (st.warning if check_p < 0.05 else st.success)(_t("incons_flag") if check_p < 0.05 else _t("incons_ok"))

    effects = st.session_state.nma_document.get("spec", {}).get("effects", "random")
    splits = incons.splits_random if effects == "random" and incons.splits_random else incons.splits_fixed
    st.markdown(f"**{_t('incons_split')}** · {'random' if splits is incons.splits_random else 'fixed'} effects")
    if not splits:
        st.info(_t("incons_none"))
    else:
        frame = sup.node_split_frame(splits, incons.measure)
        _show_table(frame.drop(columns=["_flag"]), flags=frame["_flag"].tolist())
        st.caption(_t("incons_split_hint"))
        st.download_button(_t("download_csv"), frame.drop(columns=["_flag"]).to_csv(index=False).encode("utf-8"),
                           "nma-node-split.csv", "text/csv", key="nma_split_download")
    if incons.not_splittable:
        st.caption(f"{_t('incons_not_split')}: " + ", ".join(f"{t} vs {v}" for t, v in incons.not_splittable))
    with st.expander(_t("incons_designs"), expanded=False):
        _show_table(pd.DataFrame([{
            "design": " : ".join(design.design), "studies": design.n_studies,
            "Q": round(design.q, 3), "df": design.df,
        } for design in dbt.designs]))
    _show_warnings(incons.warnings)
    _render_bayesian_inconsistency(incons)


def _render_bayesian_inconsistency(incons) -> None:
    """Bayesian node-splitting (TSD 4) and the UME comparison, when the
    Bayesian engine with the exact likelihood is in use."""

    document = st.session_state.nma_document
    spec = document.get("spec", {})
    if spec.get("engine") != "bayesian":
        return
    bayes = _active_bayes()
    st.markdown(f"### {_t('bayes_incons_title')}")
    if not isinstance(bayes, ExactResult):
        st.info(_t("bayes_incons_needs_exact"))
        return
    effects = spec.get("effects", "random")
    st.markdown(f"**{_t('bayes_split_title')}** · {effects} effects")
    st.caption(_t("bayes_split_help"))
    key = (sup.bayes_hash(document), effects)
    stored = st.session_state.get("nma_bayes_split")
    pairs = [(x.treatment, x.versus) for x in (incons.splits_random if effects == "random" and incons.splits_random
                                               else incons.splits_fixed)]
    if not pairs:
        st.info(_t("incons_none"))
    elif st.button(_t("bayes_split_run").format(n=len(pairs)), icon=":material/call_split:", key="nma_bayes_split_run"):
        try:
            model = resolve_nma(document)
        except ValueError as exc:
            st.error(str(exc))
            return
        splits = []
        progress = st.progress(0.0)
        warmup = int(spec.get("bayes", {}).get("warmup", 1000))
        for index, (a, b) in enumerate(pairs):
            run = run_nma_bayesian_exact(model, _bayes_settings(spec, effects), model_type="nodesplit",
                                         warmup=warmup, split=(a, b))
            if run.split is not None:
                splits.append(run.split)
            progress.progress((index + 1) / len(pairs), text=f"{a} vs {b}")
        progress.empty()
        stored = {"key": key, "splits": splits}
        st.session_state.nma_bayes_split = stored
    if stored:
        if stored["key"] != key:
            st.warning(_t("stale"))
        ratio = is_ratio_measure(bayes.measure)
        rows, flags = [], []
        for split in stored["splits"]:
            rows.append({
                _t("col_comparison"): f"{split.treatment} vs {split.versus}",
                _t("incons_col_direct_n"): split.n_direct_studies,
                f"{_t('direct')} ({bayes.measure})": sup.format_effect(split.direct.natural, split.direct.natural_lower,
                                                                       split.direct.natural_upper),
                f"{_t('incons_col_indirect')} ({bayes.measure})": sup.format_effect(
                    split.indirect.natural, split.indirect.natural_lower, split.indirect.natural_upper),
                (_t("incons_col_ratio") if ratio else _t("incons_col_difference")): sup.format_effect(
                    *(math.exp(v) for v in (split.difference, split.difference_lower, split.difference_upper))
                    if ratio else (split.difference, split.difference_lower, split.difference_upper)),
                "p": _fmt_p(split.p_value),
                "DIC": round(split.dic, 1) if split.dic is not None else None,
            })
            flags.append(split.p_value < 0.05)
        _show_table(pd.DataFrame(rows), flags=flags)
        st.caption(_t("bayes_split_hint").format(dic=f"{bayes.dic:.1f}"))
    _render_ume(bayes)


EE_TARGETS = {
    "markov": ("markov_document", "Markov"),
    "psm": ("psm_document", "PSM"),
    "tree": ("document", "Decision Tree"),
    "microsim": ("microsim_document", "Microsimulation"),
}


def _render_ee_body() -> None:
    result = st.session_state.get("nma_result")
    if result is None or result.fixed is None:
        st.info(_t("not_run"))
        return
    if st.session_state.get("nma_result_hash") != sup.document_hash(st.session_state.nma_document):
        st.warning(_t("stale"))
    effects_model = st.session_state.nma_document.get("spec", {}).get("effects", "random")
    bayes = _active_bayes()
    if bayes is not None:
        # Posterior mean / covariance of the log effects (normal approximation
        # of the posterior) feed the same Lognormal + correlation export.
        result = NMAResult(reference=result.reference, measure=result.measure, treatments=list(bayes.treatments),
                           fixed=bayes.fit, random=bayes.fit)

    target_col, anchor_col, prefix_col = st.columns([1, 1, 1])
    with target_col:
        target = st.selectbox(_t("ee_target"), list(EE_TARGETS), format_func=lambda key: EE_TARGETS[key][1],
                              key="nma_ee_target")
    with anchor_col:
        anchor = st.selectbox(_t("ee_anchor"), result.treatments,
                              index=result.treatments.index(result.reference), key="nma_ee_anchor")
    with prefix_col:
        several = len(st.session_state.get("nma_analyses", [])) > 1
        default_prefix = (sup.slug(sup.analysis_title(st.session_state.nma_document, st.session_state.nma_active)) + "_"
                          if several else "")
        prefix = st.text_input(_t("ee_prefix"), value=default_prefix,
                               key=f"nma_ee_prefix_{st.session_state.nma_document.get('analysis_id', '')}").strip()
    others = [name for name in result.treatments if name != anchor]
    chosen = st.multiselect(_t("ee_treatments"), others, default=others, key=f"nma_ee_treatments_{anchor}")
    add_psa = st.checkbox(_t("ee_add_psa"), value=True, key="nma_ee_add_psa")
    if not chosen:
        return
    try:
        evidence = build_evidence_set(result, effects_model, anchor=anchor, treatments=chosen, prefix=prefix,
                                      draws=bayes.draws if (bayes is not None and add_psa) else None)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown(f"**{_t('ee_preview')}** · {(_t('bayesian') + ' · ') if bayes is not None else ''}{effects_model} effects")
    _show_table(pd.DataFrame([{
        "Parameter": parameter.name,
        "Comparison": f"{parameter.treatment} vs {parameter.versus}",
        f"Base ({evidence.measure})": round(parameter.base, 4),
        "Low": round(parameter.low, 4),
        "High": round(parameter.high, 4),
        "PSA": f"{parameter.psa_type}(mean {parameter.psa_values['mean']:.4f}, SE {parameter.psa_values['se']:.4f})"
        if add_psa else "–",
    } for parameter in evidence.parameters]))
    if add_psa and evidence.correlations:
        st.caption(_t("ee_corr").format(n=len(evidence.correlations), anchor=anchor))
    if add_psa and any(parameter.samples for parameter in evidence.parameters):
        st.caption(_t("ee_empirical").format(n=len(next(p.samples for p in evidence.parameters if p.samples))))

    example = evidence.parameters[0].name
    st.markdown(_t("ee_formula").format(anchor=anchor))
    st.code(evidence.apply_formula.format(p="p_base", x=example, b="base_value"), language="python")

    doc_key, model_label = EE_TARGETS[target]
    apply_col, download_col = st.columns([1, 1])
    with apply_col:
        if st.button(_t("ee_apply").format(model=model_label), icon=":material/add_task:", type="primary",
                     key="nma_ee_apply", use_container_width=True):
            document = st.session_state.get(doc_key)
            if not isinstance(document, dict):
                st.warning(_t("ee_no_target"))
            else:
                summary = merge_into_document(document, evidence, add_psa=add_psa)
                st.session_state[doc_key] = document
                st.success(_t("ee_done").format(
                    model=model_label,
                    added=", ".join(summary["added"]) or "–",
                    updated=", ".join(summary["updated"]) or "–"))
    with download_col:
        st.download_button(_t("ee_download"), json.dumps(evidence.to_json(), indent=2, ensure_ascii=False),
                           "nma-evidence-set.json", "application/json", key="nma_ee_download",
                           use_container_width=True)


def _render_rob_sensitivity(result) -> None:
    st.markdown(f"**{_t('rob_title')}**")
    high = sup.high_rob_studies(st.session_state.nma_document.get("study_characteristics", []))
    if not high:
        st.caption(_t("rob_none"))
        return
    try:
        model = resolve_nma(st.session_state.nma_document)
    except ValueError:
        return
    effects_model = st.session_state.nma_document.get("spec", {}).get("effects", "random")
    reduced = exclude_studies(model, high)
    st.caption(_t("rob_excluded").format(ids=", ".join(high)))
    if reduced.errors:
        st.warning("; ".join(error.message for error in reduced.errors[:3]))
        return
    full_fit = result.random if effects_model == "random" else result.fixed
    reduced_fit = reduced.random if effects_model == "random" else reduced.fixed
    reduced_effects = {effect.treatment: effect for effect in reduced_fit.effects}
    null = 1.0 if is_ratio_measure(result.measure) else 0.0
    rows, flags = [], []
    for effect in full_fit.effects:
        other = reduced_effects.get(effect.treatment)
        full_sig = effect.natural_lower > null or effect.natural_upper < null
        other_sig = other is not None and (other.natural_lower > null or other.natural_upper < null)
        rows.append({
            "Comparison": f"{effect.treatment} vs {effect.versus}",
            f"All studies {result.measure}": sup.format_effect(effect.natural, effect.natural_lower, effect.natural_upper),
            f"Without high RoB {result.measure}": (sup.format_effect(other.natural, other.natural_lower, other.natural_upper)
                                                   if other else "not estimable"),
        })
        flags.append(other is None or full_sig != other_sig)
    _show_table(pd.DataFrame(rows), flags=flags)


def _render_leave_one_out(result) -> None:
    st.markdown(f"**{_t('loo_title')}**")
    try:
        model = resolve_nma(st.session_state.nma_document)
    except ValueError:
        return
    effects_model = st.session_state.nma_document.get("spec", {}).get("effects", "random")
    fit = result.random if effects_model == "random" else result.fixed
    treatments = [effect.treatment for effect in fit.effects]
    if not treatments:
        return
    treatment = st.selectbox(_t("loo_treatment"), treatments, format_func=lambda name: f"{name} vs {result.reference}",
                             key="nma_loo_treatment")
    rows = _cached("loo", (effects_model, id(result)), lambda: leave_one_out(model, result, effects_model))
    table = []
    flags = []
    for row in rows:
        effect = row.effects.get(treatment)
        table.append({
            "Study removed": row.study_id,
            f"{result.measure} (CI)": sup.format_effect(*effect) if effect else (row.error or "not estimable"),
            "τ²": "–" if row.tau2 is None else round(row.tau2, 4),
            "Conclusion changes for": ", ".join(row.changed) or "–",
        })
        flags.append(bool(row.changed))
    _show_table(pd.DataFrame(table), flags=flags, max_height=380)
    st.caption(_t("loo_hint"))
    summary = influence_summary(rows, result, treatment, effects_model)
    full = {effect.treatment: effect for effect in fit.effects}[treatment]
    if summary:
        st.caption(_t("loo_summary").format(treatment=treatment, low=f"{summary['min']:.2f}",
                                            high=f"{summary['max']:.2f}", full=f"{full.natural:.2f}"))


def _render_more_body() -> None:
    try:
        model = resolve_nma(st.session_state.nma_document)
    except ValueError as exc:
        st.error(str(exc))
        return
    cinema_tab, sof_tab, contrib_tab, rank_tab, nmr_tab, funnel_tab, thr_tab, cmp_tab, des_tab, tv_tab, sv_tab, pa_tab, dr_tab, prisma_tab = st.tabs(
        [_t("cinema_tab"), _t("sof_tab"), _t("contrib_tab"), _t("rank_tab"), _t("nmr_tab"), _t("funnel_tab"),
         _t("thr_tab"), _t("cmp_tab"), _t("des_tab"), _t("tv_tab"), _t("sv_tab"), _t("pa_tab"), _t("dr_tab"), _t("prisma_tab")])
    with sof_tab:
        _render_sof(model)
    with prisma_tab:
        _render_prisma(model)
    with rank_tab:
        _render_rank_heat()
    with cinema_tab:
        _render_cinema(model)
    with contrib_tab:
        _render_contribution(model)
    with nmr_tab:
        _render_meta_regression(model)
    with funnel_tab:
        _render_funnel(model)
    with thr_tab:
        _render_threshold(model)
    with cmp_tab:
        _render_components(model)
    with des_tab:
        _render_design(model)
    with tv_tab:
        _render_timevarying()
    with sv_tab:
        _render_survival_nma()
    with pa_tab:
        _render_population_adjustment()
    with dr_tab:
        _render_dose_response()


# --------------------------------------------------------------------------
# Outcomes (several analyses in one file)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Autosave
# --------------------------------------------------------------------------


def _autosave(data: str) -> None:
    """Write the current file content to autosave/ whenever it changed since
    the last write (skipped while the untouched example is shown)."""

    fingerprint = sup.autosave_fingerprint(json.loads(data))
    if st.session_state.get("nma_autosave_fp") == fingerprint:
        st.caption(_t("autosave_status").format(time=st.session_state.get("nma_autosave_time", "–")))
        return
    st.session_state.nma_autosave_fp = fingerprint
    documents = _all_documents()
    if len(documents) == 1 and sup.is_example_data(documents[0]) and not any(
            len(row) > 1 for row in documents[0].get("study_characteristics", [])):
        return
    sup.autosave_write(json.loads(data))
    import datetime

    st.session_state.nma_autosave_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.caption(_t("autosave_status").format(time=st.session_state.nma_autosave_time))


def _offer_autosave_restore() -> None:
    """Once per session: if an autosave exists, offer to restore it."""

    if st.session_state.get("nma_autosave_checked"):
        return
    st.session_state.nma_autosave_checked = True
    data = sup.autosave_read()
    if not data:
        return
    st.session_state.nma_autosave_pending = data


def _render_autosave_banner() -> None:
    data = st.session_state.get("nma_autosave_pending")
    if not data:
        return
    when = data.get("autosaved_at", "").replace("T", " ")
    try:
        documents, active = sup.documents_from_file(data)
        titles = ", ".join(sup.analysis_title(document, index) for index, document in enumerate(documents))
        studies = sum(len(document.get("studies", [])) for document in documents)
    except ValueError:
        st.session_state.nma_autosave_pending = None
        return
    with st.container(border=True, key="nma_autosave_banner"):
        st.markdown(f"**{_t('autosave_found')}** · {when} · {html_lib.escape(data.get('name', '') or '')} · "
                    f"{html_lib.escape(titles)} · {studies} {_t('studies').lower()}")
        restore_col, discard_col, _ = st.columns([1, 1, 3])
        with restore_col:
            if st.button(_t("autosave_restore"), type="primary", icon=":material/restore:", key="nma_autosave_restore",
                         use_container_width=True):
                _load_file(documents, active)
                st.session_state.nma_autosave_pending = None
                st.session_state.nma_autosave_fp = sup.autosave_fingerprint({k: v for k, v in data.items()
                                                                              if k != "autosaved_at"})
                st.rerun()
        with discard_col:
            if st.button(_t("autosave_discard"), icon=":material/delete:", key="nma_autosave_discard",
                         use_container_width=True):
                sup.autosave_discard()
                st.session_state.nma_autosave_pending = None
                st.rerun()


# --------------------------------------------------------------------------
# Report export (Word + Excel)
# --------------------------------------------------------------------------

REPORT_LABEL_KEYS = [key for key in nma_report.DEFAULT_LABELS]


def _prisma_auto(model) -> dict[str, tuple[str, str]]:
    """Editor suggestion for the PRISMA-NMA items, from what has been run."""

    document = st.session_state.nma_document
    characteristics = document.get("study_characteristics") or []
    has_rob = any(str(row.get("rob", "") or "").strip() for row in characteristics)
    has_chars = any(any(str(v or "").strip() for k, v in row.items() if k not in ("study", "rob")) for row in characteristics)
    incons = st.session_state.get("nma_incons_result")
    bayes = _active_bayes()
    result = st.session_state.get("nma_result")
    return nma_prisma.auto_status(
        has_characteristics=has_chars, has_rob=has_rob,
        has_inconsistency=incons is not None and not incons.errors and st.session_state.get("nma_incons_hash") == _incons_hash(),
        has_cinema=bool(document.get("cinema")) or result is not None, has_funnel=result is not None,
        bayes=bayes is not None,
        has_absolute=result is not None and result.measure in ABSOLUTE_MEASURES,
        extra_analyses=[name for name, present in (("leave-one-out", True), ("prior sensitivity (τ)", bayes is not None),
                                                   ("network meta-regression", True),
                                                   ("UME model", bool(st.session_state.get("nma_ume")))) if present])


def _render_prisma(model) -> None:
    document = st.session_state.nma_document
    st.caption(_t("prisma_intro"))
    auto = _prisma_auto(model)
    frame = nma_prisma.checklist_frame(document, auto)
    counts = nma_prisma.summary(frame)
    st.markdown(_t("prisma_summary").format(reported=counts["Reported"], partly=counts["Partly"], todo=counts["To do"],
                                            na=counts["Not applicable"], total=len(frame)))
    edited = st.data_editor(
        frame, hide_index=True, use_container_width=True, num_rows="fixed",
        key=f"nma_prisma_editor_{st.session_state.nma_editor_version}",
        disabled=["Item", "Section", "Topic", "Checklist item", "Suggested"],
        column_config={
            "Item": st.column_config.TextColumn(width="small"),
            "Section": st.column_config.TextColumn(width="small"),
            "Topic": st.column_config.TextColumn(width="medium"),
            "Checklist item": st.column_config.TextColumn(width="large"),
            "Status": st.column_config.SelectboxColumn(options=list(nma_prisma.STATUSES), width="small", required=True),
            "Suggested": st.column_config.TextColumn(width="medium"),
            "Where reported / notes": st.column_config.TextColumn(width="large"),
        })
    if nma_prisma.apply_edits(document, edited, auto):
        st.rerun()
    cols = st.columns([1, 1, 3])
    with cols[0]:
        if document.get("prisma") and st.button(_t("prisma_reset"), icon=":material/restart_alt:", key="nma_prisma_reset"):
            document.pop("prisma", None)
            st.session_state.nma_editor_version += 1
            st.rerun()
    with cols[1]:
        st.download_button(_t("prisma_download"), nma_prisma.report_frame(frame).to_csv(index=False).encode("utf-8"),
                           "prisma-nma-checklist.csv", "text/csv", key="nma_prisma_download")


def _report_labels() -> dict[str, str]:
    return {key: _t(f"rep_{key}") for key in REPORT_LABEL_KEYS if f"rep_{key}" in TEXT["en"]}


def _render_report_export(model, summary) -> None:
    result = st.session_state.get("nma_result")
    st.markdown(f"**{_t('report_title_ui')}**")
    if result is None or result.random is None or model is None or summary is None:
        st.caption(_t("report_needs_run"))
        return
    document = st.session_state.nma_document
    spec = document.get("spec", {})
    effects = spec.get("effects", "random")
    bayes = _active_bayes()
    key = (sup.document_hash(document), effects, sup.bayes_hash(document) if bayes is not None else "",
           st.session_state.get("nma_incons_hash"), json.dumps(document.get("cinema", {}), sort_keys=True, default=str),
           json.dumps(document.get("absolute", {}), sort_keys=True, default=str),
           json.dumps(document.get("prisma", {}), sort_keys=True, default=str),
           json.dumps([document.get("study_characteristics", []), document.get("characteristic_columns")],
                      sort_keys=True, default=str), st.session_state.get("language"),
           getattr(st.session_state.get("nma_threshold"), "best", None), getattr(st.session_state.get("nma_threshold"), "basis", None),
           getattr(st.session_state.get("nma_components"), "q_additive", None),
           getattr(st.session_state.get("nma_design"), "nrs", None),
           len(getattr(st.session_state.get("nma_piecewise"), "estimates", []) or []),
           (st.session_state.get("nma_survival") or {}).get("selected"),
           (st.session_state.get("nma_dose") or {}).get("selected"), getattr(st.session_state.get("nma_maic"), "estimate", None), getattr(st.session_state.get("nma_mlnmr"), "loglik", None))
    stored = st.session_state.get("nma_report_bundle")
    if st.button(_t("report_build"), icon=":material/description:", key="nma_report_build", use_container_width=True):
        with st.spinner(_t("report_building")):
            incons = st.session_state.get("nma_incons_result")
            if incons is None or st.session_state.get("nma_incons_hash") != _incons_hash():
                incons = _current_inconsistency(model)
            contribution = _cached("contribution", (effects, id(result)), lambda: contribution_matrix(
                model, effects, result.heterogeneity.tau2 if effects == "random" else 0.0))
            cinema_settings = document.get("cinema", {})
            cinema = run_cinema(model, result, effects=effects, characteristics=document.get("study_characteristics", []),
                                inconsistency=incons, contribution=contribution,
                                margin=cinema_settings.get("margin") if cinema_settings.get("measure") == spec.get("measure") else None,
                                rob_rule=cinema_settings.get("rob_rule", "average"),
                                overrides=cinema_settings.get("overrides", {}))
            fit = bayes.fit if bayes is not None else (result.random if effects == "random" else result.fixed)
            absolute, p0 = None, None
            if result.measure in ABSOLUTE_MEASURES:
                try:
                    p0 = document.get("absolute", {}).get("baseline_risk") or observed_baseline_risk(model) or 0.10
                    absolute = absolute_effects(fit, result.measure, p0, spec.get("small_values", "good"))
                except ValueError:
                    absolute = None
            ume = st.session_state.get("nma_ume")
            splits = st.session_state.get("nma_bayes_split")
            sof_settings = document.get("sof", {})
            sof_layout = sof_settings.get("layout", "reference")
            sof_frame = nma_sof.sof_frame(
                document, model, summary, fit, result.measure, sof_layout, sof_settings.get("comparator"),
                cinema if not cinema.errors else None, ranking_label="SUCRA" if bayes is not None else "P-score",
                interval_label=f"{spec.get('ci_level', 0.95):.0%} {'CrI' if bayes is not None else 'CI'}")
            sof_notes = nma_sof.footnotes(result.measure, f"{spec.get('ci_level', 0.95):.0%} {'CrI' if bayes is not None else 'CI'}",
                                          bayes is not None, not cinema.errors)
            bundle = nma_report.build_report(nma_report.ReportInput(
                document=document, model=model, summary=summary, result=result, effects=effects,
                labels=_report_labels(), bayes=bayes, inconsistency=incons,
                cinema=cinema if not cinema.errors else None, contribution=contribution, absolute=absolute,
                baseline_risk=p0, ume=(ume or {}).get("ume") if bayes is not None else None,
                bayes_splits=(splits or {}).get("splits") if bayes is not None else None,
                title=document.get("name", ""), language=st.session_state.get("language", "en"),
                effects_label=_t("random") if effects == "random" else _t("fixed"),
                sof=sof_frame, sof_notes=sof_notes, threshold=st.session_state.get("nma_threshold"),
                components=st.session_state.get("nma_components"),
                design=st.session_state.get("nma_design"),
                piecewise=st.session_state.get("nma_piecewise"),
                survival=st.session_state.get("nma_survival"),
                dose=st.session_state.get("nma_dose"),
                popadj={"maic": st.session_state.get("nma_popadj_maic"), "mlnmr": st.session_state.get("nma_popadj_mlnmr")}))
        stored = {"key": key, "bundle": bundle}
        st.session_state.nma_report_bundle = stored
    if not stored:
        return
    if stored["key"] != key:
        st.caption(_t("report_stale"))
    bundle = stored["bundle"]
    st.download_button(_t("report_word"), bundle.docx_bytes, f"{bundle.filename_stem}.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True, key="nma_report_docx")
    st.download_button(_t("report_excel"), bundle.xlsx_bytes, f"{bundle.filename_stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True, key="nma_report_xlsx")


def _render_outcomes_sidebar(document: dict[str, Any], version: int) -> None:
    analyses = st.session_state.nma_analyses
    active = st.session_state.nma_active
    with st.expander(f"{_t('outcomes')} ({len(analyses)})", expanded=len(analyses) > 1):
        st.caption(_t("outcomes_help"))
        titles = [sup.analysis_title(analysis if i != active else document, i) for i, analysis in enumerate(analyses)]
        choice = st.selectbox(_t("outcome_active"), list(range(len(analyses))), index=active,
                              format_func=lambda i: titles[i], key=f"nma_outcome_pick_{version}")
        if choice != active:
            _switch_analysis(choice)
            st.rerun()
        title = st.text_input(_t("outcome_title"), value=document.get("title") or "",
                              placeholder=titles[active], key=f"nma_outcome_title_{version}").strip()
        if title:
            document["title"] = title
        else:
            document.pop("title", None)
        with st.popover(_t("outcome_new"), icon=":material/add:", width="stretch"):
            name = st.text_input(_t("outcome_new_name"), value=_t("outcome_default").format(n=len(analyses) + 1),
                                 key=f"nma_outcome_new_name_{version}")
            copy_studies = st.checkbox(_t("outcome_copy"), value=False, key=f"nma_outcome_copy_{version}")
            if st.button(_t("outcome_add"), type="primary", key=f"nma_outcome_add_{version}", use_container_width=True):
                if not document.get("title"):
                    document["title"] = titles[active]
                _stash_active()
                st.session_state.nma_analyses.append(
                    sup.new_analysis(document, name.strip() or _t("outcome_default").format(n=len(analyses) + 1),
                                     copy_studies))
                _switch_analysis(len(st.session_state.nma_analyses) - 1)
                st.rerun()
        if len(analyses) > 1 and st.button(_t("outcome_delete"), icon=":material/delete:", use_container_width=True,
                                           key=f"nma_outcome_delete_{version}"):
            removed = analyses.pop(active)
            st.session_state.nma_analysis_cache.pop(removed.get("analysis_id"), None)
            target = sup.share_common(document, analyses[max(0, active - 1)])
            st.session_state.nma_active = max(0, active - 1)
            _load_document(target)
            for key, value in st.session_state.nma_analysis_cache.get(target.get("analysis_id"), {}).items():
                st.session_state[key] = value
            st.rerun()


# --------------------------------------------------------------------------
# CINeMA and contribution matrix
# --------------------------------------------------------------------------

LEVEL_KEYS = {"No concerns": "lvl_no", "Some concerns": "lvl_some", "Major concerns": "lvl_major",
              "Undetected": "lvl_undetected", "Suspected": "lvl_suspected", "High": "lvl_high",
              "Moderate": "lvl_moderate", "Low": "lvl_low", "Very low": "lvl_very_low"}


def _level_label(level: str) -> str:
    return _t(LEVEL_KEYS[level]) if level in LEVEL_KEYS else level


def _cached(name: str, key, compute):
    """Reuse a slow result (leave-one-out, contribution matrix, funnel) while
    the fitted data are unchanged; the cache is keyed by the document hash, so
    any data edit recomputes. Kept small (latest 24 entries)."""

    cache = st.session_state.setdefault("nma_calc_cache", {})
    full_key = (name, key, sup.document_hash(st.session_state.nma_document),
                st.session_state.nma_document.get("analysis_id"))
    if full_key not in cache:
        if len(cache) >= 24:
            cache.pop(next(iter(cache)))
        cache[full_key] = compute()
    return cache[full_key]


def _current_result():
    """The frequentist result if it matches the current data, else None (with a message)."""

    result = st.session_state.get("nma_result")
    if result is None or result.random is None:
        st.info(_t("cinema_needs_run"))
        return None
    if st.session_state.get("nma_result_hash") != sup.document_hash(st.session_state.nma_document):
        st.warning(_t("stale"))
    if _active_bayes() is not None:
        st.caption(_t("cinema_freq_note"))
    return result


def _split_method() -> str:
    return st.session_state.nma_document.get("spec", {}).get("split_method", "side")


def _incons_hash() -> str:
    return sup.document_hash(st.session_state.nma_document) + "|" + _split_method()


def _current_cinema(model, result):
    """CINeMA with the settings stored in the document (no UI)."""

    document = st.session_state.nma_document
    settings = document.get("cinema", {})
    effects = document.get("spec", {}).get("effects", "random")
    incons = _current_inconsistency(model)
    contribution = _cached("contribution", (effects, id(result)), lambda: contribution_matrix(
        model, effects, result.heterogeneity.tau2 if effects == "random" else 0.0))
    order = list(model.treatment_names)
    funnel = _cached("funnel", tuple(order), lambda: comparison_adjusted_funnel(model, order))
    margin = settings.get("margin") if settings.get("measure") == model.spec.measure else None
    cinema = run_cinema(model, result, effects=effects, characteristics=document.get("study_characteristics", []),
                        inconsistency=incons, contribution=contribution, margin=margin,
                        rob_rule=settings.get("rob_rule", "average"), overrides=settings.get("overrides", {}),
                        funnel=funnel)
    return None if cinema.errors else cinema


def _sof_settings(document: dict) -> dict:
    return document.setdefault("sof", {})


def _render_sof(model) -> None:
    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    summary = summarize_network(model)
    st.caption(_t("sof_intro"))
    settings = _sof_settings(document)
    layout_labels = {"reference": _t("sof_layout_reference").format(ref=model.reference),
                     "all": _t("sof_layout_all"), "comparator": _t("sof_layout_comparator")}
    cols = st.columns([2, 1.4])
    with cols[0]:
        layout = st.segmented_control(_t("sof_layout"), list(nma_sof.LAYOUTS), default=settings.get("layout", "reference"),
                                      format_func=lambda key: layout_labels[key], key="nma_sof_layout") or "reference"
    comparator = None
    with cols[1]:
        if layout == "comparator":
            names = list(model.treatment_names)
            stored = settings.get("comparator") if settings.get("comparator") in names else names[0]
            comparator = st.selectbox(_t("sof_comparator"), names, index=names.index(stored), key="nma_sof_comparator")
    settings.update({"layout": layout, "comparator": comparator or settings.get("comparator")})
    bayes = _active_bayes()
    fit = bayes.fit if bayes is not None else (result.random if document.get("spec", {}).get("effects", "random") == "random"
                                               else result.fixed)
    cinema = _current_cinema(model, result)
    interval = f"{document.get('spec', {}).get('ci_level', 0.95):.0%} {'CrI' if bayes is not None else 'CI'}"
    frame = nma_sof.sof_frame(document, model, summary, fit, result.measure, layout, comparator, cinema,
                              ranking_label="SUCRA" if bayes is not None else "P-score", interval_label=interval)
    _show_table(frame)
    for note in nma_sof.footnotes(result.measure, interval, bayes is not None, cinema is not None):
        st.caption(note)
    st.download_button(_t("download_csv"), frame.to_csv(index=False).encode("utf-8"), "nma-summary-of-findings.csv",
                       "text/csv", key="nma_sof_download")


def _current_inconsistency(model):
    current = _incons_hash()
    if st.session_state.get("nma_incons_result") is None or st.session_state.get("nma_incons_hash") != current:
        st.session_state.nma_incons_result = run_inconsistency(model, method=_split_method())
        st.session_state.nma_incons_hash = current
    return st.session_state.nma_incons_result


def _render_cinema(model) -> None:
    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    analysis_id = document.get("analysis_id", "")
    settings = document.setdefault("cinema", {})
    measure = model.spec.measure
    ratio = is_ratio_measure(measure)
    effects = document.get("spec", {}).get("effects", "random")
    st.caption(_t("cinema_intro"))

    margin_col, rule_col = st.columns([1, 1.2])
    with margin_col:
        stored = settings.get("margin") if settings.get("measure") == measure else None
        margin = st.number_input(
            _t("cinema_margin_ratio" if ratio else "cinema_margin_diff").format(measure=measure),
            min_value=1.0 if ratio else 0.0, step=0.05 if ratio else 0.1, format="%.3f",
            value=float(stored if stored is not None else default_margin(measure)),
            help=_t("cinema_margin_ratio_help" if ratio else "cinema_margin_diff_help"),
            key=f"nma_cinema_margin_{analysis_id}_{measure}")
    with rule_col:
        rule_labels = {"average": _t("cinema_rule_average"), "majority": _t("cinema_rule_majority")}
        rule = st.segmented_control(_t("cinema_rule"), ["average", "majority"],
                                    default=settings.get("rob_rule", "average"), help=_t("cinema_rule_help"),
                                    format_func=lambda key: rule_labels[key],
                                    key=f"nma_cinema_rule_{analysis_id}") or "average"
    settings.update({"margin": margin, "measure": measure, "rob_rule": rule})
    overrides = settings.setdefault("overrides", {})
    incons = _current_inconsistency(model)
    contribution = _cached("contribution", (effects, id(result)), lambda: contribution_matrix(
        model, effects, result.heterogeneity.tau2 if effects == "random" else 0.0))

    order = list(model.treatment_names)
    funnel = _cached("funnel", tuple(order), lambda: comparison_adjusted_funnel(model, order))

    def evaluate():
        return run_cinema(model, result, effects=effects, characteristics=document.get("study_characteristics", []),
                          inconsistency=incons, contribution=contribution, margin=margin, rob_rule=rule,
                          overrides=overrides, funnel=funnel)

    cinema = evaluate()
    if cinema.errors:
        st.warning("\n".join(getattr(error, "message", str(error)) for error in cinema.errors))
        return
    table_slot = st.container()

    with st.expander(_t("cinema_edit"), expanded=False):
        labels = {row.key: row.label for row in cinema.rows}
        chosen_key = st.selectbox(_t("cinema_comparison"), list(labels), format_func=lambda key: labels[key],
                                  key=f"nma_cinema_pick_{analysis_id}")
        row = next(r for r in cinema.rows if r.key == chosen_key)
        chosen = dict(overrides.get(chosen_key, {}))
        grid = st.columns(3)
        for i, domain in enumerate(DOMAINS):
            judgement = row.domains[domain]
            levels = REPORTING_LEVELS if domain == "reporting_bias" else CONCERN_LEVELS
            options = ["auto", *levels]
            current = chosen.get(domain, "auto")
            with grid[i % 3]:
                picked = st.selectbox(
                    _t(f"dom_{domain}"), options, index=options.index(current) if current in options else 0,
                    format_func=lambda value, auto=judgement.auto_level: (
                        _t("cinema_auto").format(level=_level_label(auto)) if value == "auto" else _level_label(value)),
                    key=f"nma_cinema_{analysis_id}_{chosen_key}_{domain}")
                st.caption(judgement.reason)
            if picked == "auto":
                chosen.pop(domain, None)
            else:
                chosen[domain] = picked
        confidence_options = ["auto", *reversed(CONFIDENCE)]
        confidence_col, note_col = st.columns([1, 2])
        with confidence_col:
            current = chosen.get("confidence", "auto")
            picked = st.selectbox(
                _t("col_confidence"), confidence_options,
                index=confidence_options.index(current) if current in confidence_options else 0,
                format_func=lambda value: (_t("cinema_auto").format(level=_level_label(row.auto_confidence))
                                           if value == "auto" else _level_label(value)),
                key=f"nma_cinema_{analysis_id}_{chosen_key}_confidence")
        if picked == "auto":
            chosen.pop("confidence", None)
        else:
            chosen["confidence"] = picked
        with note_col:
            note = st.text_input(_t("cinema_note"), value=chosen.get("note", ""),
                                 key=f"nma_cinema_{analysis_id}_{chosen_key}_note").strip()
        if note:
            chosen["note"] = note
        else:
            chosen.pop("note", None)
        if chosen:
            overrides[chosen_key] = chosen
        else:
            overrides.pop(chosen_key, None)
        if overrides.get(chosen_key) and st.button(_t("cinema_reset"), icon=":material/restart_alt:",
                                                   key=f"nma_cinema_reset_{analysis_id}_{chosen_key}"):
            overrides.pop(chosen_key, None)
            for suffix in (*DOMAINS, "confidence", "note"):
                st.session_state.pop(f"nma_cinema_{analysis_id}_{chosen_key}_{suffix}", None)
            st.rerun()
    cinema = evaluate()

    st.session_state.nma_cinema_summary = {row.label: row.confidence for row in cinema.rows}
    with table_slot:
        domain_labels = {domain: _t(f"dom_{domain}") for domain in DOMAINS}
        level_labels = {level: _level_label(level) for level in LEVEL_KEYS}
        headers = {"comparison": _t("col_comparison"), "estimate": f"{measure} ({_t('col_estimate')})",
                   "pi": _t("col_pi"), "direct": _t("col_direct"), "confidence": _t("col_confidence")}
        st.markdown(sup.cinema_html_table(cinema, domain_labels, level_labels, headers), unsafe_allow_html=True)
        _show_warnings(cinema.warnings)
        report = pd.DataFrame(cinema_report_rows(cinema))
        st.download_button(_t("cinema_download"), report.to_csv(index=False).encode("utf-8"),
                           f"nma-cinema-{sup.slug(sup.analysis_title(document, st.session_state.nma_active))}.csv",
                           "text/csv", key=f"nma_cinema_download_{analysis_id}")
    with st.expander(_t("cinema_rules"), expanded=False):
        st.markdown(_t("cinema_rules_text"))


ROB_COLORS = {"Low": "#16a34a", "Some concerns": "#f59e0b", "High": "#dc2626", "": "#94a3b8"}


def _render_contribution(model) -> None:
    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    effects = document.get("spec", {}).get("effects", "random")
    contribution = _cached("contribution", (effects, id(result)), lambda: contribution_matrix(
        model, effects, result.heterogeneity.tau2 if effects == "random" else 0.0))
    if contribution.errors:
        st.warning("\n".join(error.message for error in contribution.errors))
        return
    st.caption(_t("contrib_intro"))
    st.markdown(f"`{_t('random') if effects == 'random' else _t('fixed')} effects`"
                + (f" · `τ² = {contribution.tau2:.3f}`" if effects == "random" else ""))
    st.markdown(sup.contribution_html_table(contribution, {"estimate": _t("contrib_estimate")}),
                unsafe_allow_html=True)
    st.download_button(_t("contrib_download"),
                       sup.contribution_frame(contribution).to_csv(index=False).encode("utf-8"),
                       "nma-contribution-matrix.csv", "text/csv", key="nma_contrib_download")

    labels = {f"{row.treatment}|{row.versus}": row.label for row in contribution.rows}
    key = st.selectbox(_t("contrib_studies"), list(labels), format_func=lambda k: labels[k], key="nma_contrib_pick")
    row = next(r for r in contribution.rows if f"{r.treatment}|{r.versus}" == key)
    rob = {str(item.get("study")): normalize_level(item.get("rob"))
           for item in document.get("study_characteristics", []) if item.get("study")}
    frame = pd.DataFrame([{"study": study_id, "share": value * 100, "rob": rob.get(study_id, "")}
                          for study_id, value in row.studies.items() if value > 0.0005])
    if frame.empty:
        return
    frame = frame.sort_values("share", ascending=False)
    chart = alt.Chart(frame).mark_bar().encode(
        y=alt.Y("study:N", sort=list(frame["study"]), title=_t("contrib_study")),
        x=alt.X("share:Q", title=_t("contrib_share")),
        color=alt.Color("rob:N", title=_t("contrib_rob"),
                        scale=alt.Scale(domain=list(ROB_COLORS), range=list(ROB_COLORS.values())),
                        legend=alt.Legend(orient="bottom", labelExpr="datum.label || '–'")),
        tooltip=["study", alt.Tooltip("share:Q", format=".1f"), "rob"],
    ).properties(height=max(120, 26 * len(frame)))
    st.altair_chart(chart, use_container_width=True)


def _render_threshold(model) -> None:
    """B2: threshold analysis (Phillippo 2018) on the effect or the net-monetary-benefit basis."""

    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    effects = document.get("spec", {}).get("effects", "random")
    st.caption(_t("thr_intro"))
    used = list(result.treatments) or list(model.treatment_names)
    basis = st.radio(_t("thr_basis"), ["effect", "nmb"], horizontal=True, key="nma_thr_basis",
                     format_func=lambda v: _t("thr_basis_effect") if v == "effect" else _t("thr_basis_nmb"))
    decision = nma_threshold.DecisionInputs(basis=basis)
    if basis == "nmb":
        st.caption(_t("thr_nmb_help"))
        decision.wtp = float(st.number_input(_t("thr_wtp"), min_value=0.0, value=30000.0, step=1000.0, key="nma_thr_wtp"))
        frame = pd.DataFrame({"treatment": used, "cost": 0.0, "effect": 0.0, "d_effect": 0.0, "d_cost": 0.0})
        edited = st.data_editor(
            frame, hide_index=True, use_container_width=True, key=f"nma_thr_values_{'|'.join(used)}",
            disabled=["treatment"], column_config={
                "treatment": st.column_config.TextColumn(_t("thr_col_treatment")),
                "cost": st.column_config.NumberColumn(_t("thr_col_cost"), format="%g"),
                "effect": st.column_config.NumberColumn(_t("thr_col_effect"), format="%g"),
                "d_effect": st.column_config.NumberColumn(_t("thr_col_d_effect"), format="%g", help=_t("thr_col_d_effect_help")),
                "d_cost": st.column_config.NumberColumn(_t("thr_col_d_cost"), format="%g", help=_t("thr_col_d_cost_help"))})
        decision.values = {str(r["treatment"]): {k: float(r[k]) if r[k] == r[k] and r[k] is not None else 0.0
                                                 for k in ("cost", "effect", "d_effect", "d_cost")}
                           for _, r in edited.iterrows()}
    out = _cached("threshold", (effects, basis, decision.wtp, json.dumps(decision.values, sort_keys=True), id(result)),
                  lambda: nma_threshold.run_threshold_analysis(model, effects, decision, result))
    st.session_state.nma_threshold = out
    if out.errors:
        st.warning("\n".join(error.message for error in out.errors))
        return
    for message in out.warnings:
        st.warning(message)
    st.markdown(_t("thr_best").format(best=out.best, order=" > ".join(out.order)))
    if basis == "nmb" and out.icer:
        st.markdown(f"**{_t('thr_icer_title')}**")
        st.caption(_t("thr_icer_help"))
        icer = pd.DataFrame(out.icer).rename(columns={
            "treatment": _t("thr_col_treatment"), "cost": _t("thr_col_cost"), "effect": _t("thr_col_effect"),
            "incremental_cost": _t("thr_inc_cost"), "incremental_effect": _t("thr_inc_effect"), "icer": "ICER", "nmb": "NMB"})
        st.dataframe(icer, hide_index=True, use_container_width=True)
    scope = st.radio(_t("thr_scope"), ["studies", "network"], horizontal=True, key="nma_thr_scope",
                     format_func=lambda v: _t("thr_scope_studies") if v == "studies" else _t("thr_scope_network"))
    rows = out.rows if scope == "studies" else out.network_rows
    if not rows:
        return
    labels = {"study": _t("thr_study"), "comparison": _t("thr_comparison"), "estimate": _t("thr_estimate"),
              "lower": _t("thr_lower"), "lower_best": _t("thr_lower_best"), "upper": _t("thr_upper"),
              "upper_best": _t("thr_upper_best"), "fragile": _t("thr_fragile"), "none": _t("thr_never"),
              "yes": _t("yes"), "no": _t("no")}
    table = sup.threshold_frame(out, scope, labels)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"),
                       "nma-threshold-analysis.csv", "text/csv", key="nma_thr_download")
    chart_rows = []
    for r in rows:
        name = f"{r.study}: {r.comparison}"
        chart_rows.append({"label": name, "estimate": r.natural, "lower": r.natural_lower, "upper": r.natural_upper,
                           "fragile": _t("thr_fragile_yes") if r.fragile else _t("thr_fragile_no")})
    chart_df = pd.DataFrame(chart_rows)
    log_axis = is_ratio_measure(model.spec.measure)
    scale = alt.Scale(type="log") if log_axis else alt.Scale(zero=False)
    color = alt.Color("fragile:N", title=None, scale=alt.Scale(
        domain=[_t("thr_fragile_no"), _t("thr_fragile_yes")], range=["#2b6cb0", "#c53030"]))
    y = alt.Y("label:N", sort=list(chart_df["label"]), title=None)
    lower = alt.Chart(chart_df.dropna(subset=["lower"])).mark_rule(strokeWidth=3).encode(
        y=y, x=alt.X("lower:Q", scale=scale, title=_t("thr_axis")), x2="estimate:Q", color=color)
    upper = alt.Chart(chart_df.dropna(subset=["upper"])).mark_rule(strokeWidth=3).encode(
        y=y, x=alt.X("estimate:Q", scale=scale), x2="upper:Q", color=color)
    dots = alt.Chart(chart_df).mark_point(filled=True, size=60, color="black").encode(
        y=y, x=alt.X("estimate:Q", scale=scale), tooltip=["label", "estimate", "lower", "upper"])
    st.altair_chart((lower + upper + dots).properties(height=max(120, 24 * len(chart_df))), use_container_width=True)
    st.caption(_t("thr_chart_help"))



def _render_components(model) -> None:
    """B3: additive component NMA (Rücker 2020)."""

    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    effects = document.get("spec", {}).get("effects", "random")
    st.caption(_t("cmp_intro"))
    saved = document.get("components", {}) or {}
    names = list(result.treatments) or list(model.treatment_names)
    frame = pd.DataFrame({"treatment": names, "components": [saved.get(n, "") for n in names]})
    edited = st.data_editor(
        frame, hide_index=True, use_container_width=True, key=f"nma_cmp_editor_{'|'.join(names)}", disabled=["treatment"],
        column_config={"treatment": st.column_config.TextColumn(_t("thr_col_treatment")),
                       "components": st.column_config.TextColumn(_t("cmp_col_components"), help=_t("cmp_col_help"))})
    mapping = {str(r["treatment"]): str(r["components"] or "").strip() for _, r in edited.iterrows()}
    mapping = {k: v for k, v in mapping.items() if v}
    if mapping != saved:
        document["components"] = mapping
    if not mapping:
        st.info(_t("cmp_none"))
        return
    if document.get("spec", {}).get("component_model") == "bayes":
        _render_components_bayes(model, document, mapping, effects, result)
        return
    st.caption(_t("cmp_settings_hint"))
    out = _cached("components", (effects, json.dumps(mapping, sort_keys=True), id(result)),
                  lambda: nma_components.run_component_nma(model, mapping, effects))
    st.session_state.nma_components = out
    if out.errors:
        st.warning("\n".join(error.message for error in out.errors))
        return
    for message in out.warnings:
        st.warning(message)
    interval = f"{document.get('spec', {}).get('ci_level', 0.95):.0%} CI"
    st.markdown(f"**{_t('cmp_effects_title')}**")
    st.dataframe(sup.component_effects_frame(out, interval, {"component": _t("cmp_component"), "effect": _t("thr_estimate")}),
                 hide_index=True, use_container_width=True)
    st.markdown(f"**{_t('cmp_treatments_title')}**")
    labels = {"treatment": _t("thr_col_treatment"), "components": _t("cmp_col_components"),
              "additive": _t("cmp_additive"), "standard": _t("cmp_standard")}
    table = sup.component_treatments_frame(out, labels)
    st.dataframe(table, hide_index=True, use_container_width=True)
    p = "–" if out.p_additivity is None else f"{out.p_additivity:.3f}"
    st.caption(_t("cmp_additivity").format(q=f"{out.q_diff:.2f}", df=out.df_diff, p=p))
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"),
                       "nma-component-effects.csv", "text/csv", key="nma_cmp_download")



def _render_components_bayes(model, document: dict, mapping: dict, effects: str, result) -> None:
    """Bayesian additive component NMA (+ optional interaction terms), same sampler as the standard Bayesian NMA."""

    from heor_tree_engine.nma_components_bayes import BayesComponentSettings, run_component_nma_bayes

    parts: list[str] = []
    for name in list(result.treatments):
        for part in nma_components.parse_components(mapping.get(name)) or ([name] if name != model.reference else []):
            if part not in parts:
                parts.append(part)
    spec = document.get("spec", {})
    pairs = {f"{a}:{b}" for i, a in enumerate(parts) for b in parts[i + 1:]}
    chosen = [item for item in spec.get("component_interactions", []) if item in pairs]
    if chosen:
        st.caption(_t("cmp_interactions_used").format(items=", ".join(chosen)))
    settings = BayesComponentSettings(
        bayes=_bayes_settings(spec, effects),
        interactions=[tuple(item.split(":", 1)) for item in chosen],
    )
    key = (effects, json.dumps(mapping, sort_keys=True), tuple(chosen), json.dumps(spec.get("bayes", {}), sort_keys=True), id(result))
    with st.spinner("MCMC..."):
        out = _cached("components_bayes", key, lambda: run_component_nma_bayes(model, mapping, settings))
    if out.errors:
        st.warning("\n".join(error.message for error in out.errors))
        return
    for message in out.warnings:
        st.warning(message)
    interval = f"{spec.get('ci_level', 0.95):.0%} CrI"
    st.markdown(f"**{_t('cmp_effects_title')}**")
    st.dataframe(sup.component_effects_frame(out, interval, {"component": _t("cmp_component"), "effect": _t("thr_estimate")}),
                 hide_index=True, use_container_width=True)
    st.markdown(f"**{_t('cmp_bayes_title')}**")
    rows = [{
        _t("thr_col_treatment"): e.treatment,
        _t("cmp_col_components"): " + ".join(out.membership.get(e.treatment, [])),
        _t("cmp_additive"): f"{e.natural:.3f} ({e.natural_lower:.3f} to {e.natural_upper:.3f})",
        "SUCRA": round(out.sucra.get(e.treatment, float("nan")), 3),
    } for e in out.fit.effects]
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    with st.expander(_t("cmp_bayes_diag")):
        st.dataframe(pd.DataFrame([{"parameter": d.name, "median": round(d.median, 4), "sd": round(d.sd, 4),
                                    "R-hat": round(d.rhat, 3), "ESS": round(d.ess)} for d in out.diagnostics]),
                     hide_index=True, use_container_width=True)
        if any(d.rhat > 1.05 for d in out.diagnostics):
            st.warning("R-hat > 1.05: increase the number of draws in the MCMC settings.")
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"),
                       "nma-component-effects-bayes.csv", "text/csv", key="nma_cmp_bayes_download")


def _render_design(model) -> None:
    """B4: sensitivity to non-randomised studies."""

    result = _current_result()
    if result is None:
        return
    document = st.session_state.nma_document
    effects = document.get("spec", {}).get("effects", "random")
    st.caption(_t("des_intro"))
    ids = [study.id for study in model.studies]
    saved = set(document.get("non_randomised", []) or [])
    frame = pd.DataFrame({"study": ids, "nrs": [i in saved for i in ids]})
    edited = st.data_editor(frame, hide_index=True, use_container_width=True, key=f"nma_des_editor_{'|'.join(ids)}",
                            disabled=["study"], column_config={
                                "study": st.column_config.TextColumn(_t("thr_study")),
                                "nrs": st.column_config.CheckboxColumn(_t("des_col_nrs"))})
    chosen = [str(r["study"]) for _, r in edited.iterrows() if r["nrs"]]
    if set(chosen) != saved:
        document["non_randomised"] = chosen
    if not chosen:
        st.info(_t("des_none"))
        return
    alphas = tuple(sorted({float(v) for v in st.multiselect(
        _t("des_alphas"), [0.75, 0.5, 0.25, 0.1], default=[0.5, 0.25, 0.1], key="nma_des_alphas")}, reverse=True))
    out = _cached("design", (effects, tuple(chosen), alphas, id(result)),
                  lambda: nma_design.run_design_sensitivity(model, chosen, alphas, effects))
    st.session_state.nma_design = out
    if out.errors:
        st.warning("\n".join(error.message for error in out.errors))
        return
    for message in out.warnings:
        st.warning(message)
    table = sup.design_frame(out, {"treatment": _t("thr_col_treatment"), "studies": _t("thr_study")})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(_t("des_note"))
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"),
                       "nma-nonrandomised-sensitivity.csv", "text/csv", key="nma_des_download")



PIECEWISE_COLUMNS = ["study", "treatment", "versus", "t_from", "t_to", "log_hr", "se", "hr", "lower", "upper", "baseline_se"]


def _render_timevarying() -> None:
    """C1a/C1b: piecewise hazard ratios by time interval and proportional-hazards checks."""

    document = st.session_state.nma_document
    effects = document.get("spec", {}).get("effects", "random")
    st.caption(_t("tv_intro"))
    saved = document.get("survival_hr") or []
    frame = pd.DataFrame(saved, columns=PIECEWISE_COLUMNS) if saved else pd.DataFrame(
        {c: pd.Series(dtype="float64" if c not in ("study", "treatment", "versus") else "object") for c in PIECEWISE_COLUMNS})
    edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, use_container_width=True, key="nma_tv_editor",
                            column_config={"t_from": st.column_config.NumberColumn(_t("tv_from")),
                                           "t_to": st.column_config.NumberColumn(_t("tv_to")),
                                           "log_hr": st.column_config.NumberColumn("log HR"),
                                           "hr": st.column_config.NumberColumn("HR"),
                                           "lower": st.column_config.NumberColumn(_t("tv_lower")),
                                           "upper": st.column_config.NumberColumn(_t("tv_upper")),
                                           "baseline_se": st.column_config.NumberColumn(_t("tv_baseline_se"), help=_t("tv_baseline_help"))})
    rows = json.loads(edited.to_json(orient="records"))
    if rows != saved:
        document["survival_hr"] = rows
    clean, _ = nma_timevarying.normalize_rows(rows)
    if not clean:
        st.info(_t("tv_none"))
        return
    names = []
    for r in clean:
        for n in (r["versus"], r["treatment"]):
            if n not in names:
                names.append(n)
    reference = st.selectbox(_t("tv_reference"), names, index=names.index(clean[0]["versus"]), key="nma_tv_reference")
    unit = st.text_input(_t("tv_unit"), value="months", key="nma_tv_unit")
    out = nma_timevarying.run_piecewise_nma(rows, reference, effects)
    st.session_state.nma_piecewise = out
    for message in out.warnings:
        st.warning(message)
    if out.errors:
        st.warning("\n".join(e.message for e in out.errors))
        return
    table = sup.piecewise_frame(out, unit, {"treatment": _t("thr_col_treatment"), "studies": _t("thr_study")})
    st.markdown(f"**{_t('tv_results')}**")
    st.dataframe(table, hide_index=True, use_container_width=True)
    plot = pd.DataFrame([{"treatment": e.treatment, "t_from": e.t_from, "t_to": e.t_to, "hr": e.hr, "lower": e.lower,
                          "upper": e.upper} for e in out.estimates])
    base = alt.Chart(plot).encode(color=alt.Color("treatment:N", title=None), y=alt.Y("hr:Q", scale=alt.Scale(type="log"), title="HR"))
    chart = base.mark_rule(strokeWidth=3).encode(
        x=alt.X("t_from:Q", title=unit), x2="t_to:Q", tooltip=["treatment", "hr", "lower", "upper"])
    st.altair_chart(chart.properties(height=260), use_container_width=True)
    st.caption(_t("tv_chart_help"))
    if out.ph:
        st.markdown(f"**{_t('tv_ph_title')}**")
        st.caption(_t("tv_ph_help"))
        ph = sup.ph_frame(out, {"scope": _t("tv_level"), "item": _t("tv_item"), "intervals": _t("tv_intervals"),
                                "flag": _t("tv_flag"), "study": _t("thr_study"), "network": _t("thr_scope_network"),
                                "yes": _t("tv_possibly"), "no": _t("tv_no_evidence")})
        st.dataframe(ph, hide_index=True, use_container_width=True)
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"),
                       "nma-piecewise-hr.csv", "text/csv", key="nma_tv_download")



SURVIVAL_ARM_COLUMNS = ["study", "treatment", "t_from", "t_to", "events", "person_time"]
KM_POINT_COLUMNS = ["study", "treatment", "time", "survival"]
KM_RISK_COLUMNS = ["study", "treatment", "time", "at_risk"]


def _simple_table_editor(document: dict, key: str, columns: list[str], editor_key: str, text_columns=("study", "treatment")):
    saved = document.get(key) or []
    frame = pd.DataFrame(saved, columns=columns) if saved else pd.DataFrame(
        {c: pd.Series(dtype="object" if c in text_columns else "float64") for c in columns})
    edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, use_container_width=True, key=editor_key)
    rows = json.loads(edited.to_json(orient="records"))
    if rows != saved:
        document[key] = rows
    return rows


def _render_survival_nma() -> None:
    """C1c: parametric / fractional-polynomial survival NMA on interval counts, optionally reconstructed from KM curves."""

    document = st.session_state.nma_document
    st.caption(_t("sv_intro"))
    st.markdown(f"**{_t('sv_data_title')}**")
    st.caption(_t("sv_data_help"))
    rows = _simple_table_editor(document, "survival_arms", SURVIVAL_ARM_COLUMNS, "nma_sv_arms")
    with st.expander(_t("sv_km_title")):
        st.caption(_t("sv_km_help"))
        st.markdown(_t("sv_km_points"))
        points = _simple_table_editor(document, "km_points", KM_POINT_COLUMNS, "nma_sv_km_points")
        st.markdown(_t("sv_km_risk"))
        risk = _simple_table_editor(document, "km_risk", KM_RISK_COLUMNS, "nma_sv_km_risk")
        breaks_text = st.text_input(_t("sv_breaks"), value=document.get("km_breaks", "0, 3, 6, 9, 12, 18, 24, 36"), key="nma_sv_breaks")
        document["km_breaks"] = breaks_text
        if st.button(_t("sv_reconstruct"), key="nma_sv_reconstruct"):
            try:
                breaks = [float(x) for x in breaks_text.replace(";", ",").split(",") if x.strip()]
                if len(breaks) < 2 or breaks != sorted(breaks) or breaks[0] != 0:
                    raise ValueError(_t("sv_breaks_error"))
            except ValueError as exc:
                st.error(str(exc))
            else:
                new_rows, notes = [], []
                keys = sorted({(str(r["study"]), str(r["treatment"])) for r in points if r.get("study") and r.get("treatment")})
                for study, treatment in keys:
                    curve = [(float(r["time"]), float(r["survival"])) for r in points
                             if (str(r["study"]), str(r["treatment"])) == (study, treatment)
                             and r.get("time") is not None and r.get("survival") is not None]
                    table = sorted((float(r["time"]), float(r["at_risk"])) for r in risk
                                   if (str(r["study"]), str(r["treatment"])) == (study, treatment)
                                   and r.get("time") is not None and r.get("at_risk") is not None)
                    arm = nma_km_reconstruct.reconstruct_arm(curve, [t for t, _ in table], [n for _, n in table])
                    notes.extend(f"{study} / {treatment}: {w}" for w in arm.warnings)
                    if arm.n:
                        for a, b, e, py in nma_km_reconstruct.interval_counts(arm, breaks):
                            new_rows.append({"study": study, "treatment": treatment, "t_from": a, "t_to": b,
                                             "events": e, "person_time": round(py, 3)})
                replaced = {(r["study"], r["treatment"]) for r in new_rows}
                document["survival_arms"] = [r for r in (document.get("survival_arms") or [])
                                             if (str(r.get("study")), str(r.get("treatment"))) not in replaced] + new_rows
                st.session_state["nma_sv_notes"] = notes
                st.rerun()
        for note in st.session_state.get("nma_sv_notes", []):
            st.warning(note)
    arms = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        try:
            if r.get("study") and r.get("treatment") and None not in (r.get("t_from"), r.get("t_to"), r.get("events"), r.get("person_time")):
                groups.setdefault((str(r["study"]), str(r["treatment"])), []).append(r)
        except (TypeError, ValueError):
            continue
    for (study, treatment), items in groups.items():
        arms.append(nma_survival.SurvivalArmData(study, treatment, [float(i["t_from"]) for i in items], [float(i["t_to"]) for i in items],
                                                 [float(i["events"]) for i in items], [float(i["person_time"]) for i in items]))
    if len(arms) < 2:
        st.info(_t("sv_none"))
        return
    names = []
    for a in arms:
        if a.treatment not in names:
            names.append(a.treatment)
    left, right = st.columns(2)
    reference = left.selectbox(_t("tv_reference"), names, key="nma_sv_reference")
    effects = right.radio(_t("effects_model"), ["fixed", "random"], horizontal=True, key="nma_sv_effects",
                          index=0 if document.get("spec", {}).get("effects") == "fixed" else 1,
                          format_func=lambda v: _t(v))
    unit = left.text_input(_t("tv_unit"), value="months", key="nma_sv_unit")
    factor = right.number_input(_t("sv_time_factor"), value=1.0 / 12.0, format="%g", key="nma_sv_factor", help=_t("sv_time_factor_help"))
    model_choice = st.selectbox(_t("sv_model"), ["exponential", "weibull", "gompertz", "fp1", "fp2"], index=1, key="nma_sv_model",
                                format_func=lambda m: {"exponential": "Exponential", "weibull": "Weibull", "gompertz": "Gompertz",
                                                       "fp1": "Fractional polynomial, order 1", "fp2": "Fractional polynomial, order 2"}[m])
    powers = []
    if model_choice in ("fp1", "fp2"):
        n_pow = 1 if model_choice == "fp1" else 2
        cols = st.columns(n_pow)
        for i in range(n_pow):
            powers.append(float(cols[i].selectbox(f"p{i + 1}", list(nma_survival.FP_POWERS), index=3 if i == 0 else 5, key=f"nma_sv_p{i}")))
    c1, c2, c3 = st.columns(3)
    chains = int(c1.number_input(_t("chains"), 2, 6, 3, key="nma_sv_chains"))
    warm = int(c2.number_input(_t("sv_warmup"), 200, 5000, 800, step=100, key="nma_sv_warm"))
    draws = int(c3.number_input(_t("sv_draws"), 200, 5000, 800, step=100, key="nma_sv_draws"))
    stored = st.session_state.setdefault("nma_survival", {"fits": {}, "selected": None})

    def fit_one(name: str, pw: list[float]):
        result = nma_survival.run_survival_nma(arms, reference, nma_survival.SurvivalSettings(
            model=name, powers=pw, effects=effects, chains=chains, warmup=warm, draws_per_chain=draws))
        label = {"exponential": "Exponential", "weibull": "Weibull", "gompertz": "Gompertz"}.get(name, name.upper() + " (" + ", ".join(f"{p:g}" for p in pw) + ")")
        return label, result

    b1, b2 = st.columns(2)
    run_one = b1.button(_t("sv_run"), type="primary", key="nma_sv_run")
    run_all = b2.button(_t("sv_run_all"), key="nma_sv_run_all")
    if run_one or run_all:
        todo = [(model_choice, powers)] if run_one else [("exponential", []), ("weibull", []), ("gompertz", [])]
        with st.spinner(_t("sv_running")):
            for name, pw in todo:
                label, result = fit_one(name, pw)
                if result.errors:
                    st.error("\n".join(e.message for e in result.errors))
                    continue
                stored["fits"][label] = result
                stored["selected"] = label
    fits = stored["fits"]
    if not fits:
        return
    st.markdown(f"**{_t('sv_compare')}**")
    st.dataframe(sup.survival_models_frame(fits, {"model": _t("sv_model"), "effects": _t("effects_model"), "fixed": _t("fixed"), "random": _t("random")}),
                 hide_index=True, use_container_width=True)
    labels = list(fits)
    selected = st.selectbox(_t("sv_show"), labels, index=labels.index(stored["selected"]) if stored["selected"] in labels else 0, key="nma_sv_show")
    stored["selected"] = selected
    result = fits[selected]
    for message in result.warnings:
        st.warning(message)
    st.markdown(f"**{_t('sv_effects_title')}**")
    st.caption(_t("sv_effects_help"))
    st.dataframe(sup.survival_effects_frame(result), hide_index=True, use_container_width=True)
    studies = sorted({s for (s, t) in result.theta_draws})
    baseline = st.selectbox(_t("sv_baseline"), ["pooled"] + studies, key="nma_sv_baseline",
                            format_func=lambda v: _t("sv_baseline_pooled") if v == "pooled" else v)
    horizon = st.number_input(_t("sv_horizon"), min_value=float(result.max_time), value=float(result.max_time) * 2.0, key="nma_sv_horizon")
    grid = np.linspace(0.0, float(horizon), 61)
    table = nma_survival.survival_table(result, grid, baseline)
    chart_base = alt.Chart(table).encode(x=alt.X("time:Q", title=unit), color=alt.Color("treatment:N", title=None))
    band = chart_base.mark_area(opacity=0.15).encode(y=alt.Y("lower:Q", title=_t("sv_survival"), scale=alt.Scale(domain=[0, 1])), y2="upper:Q")
    line = chart_base.mark_line().encode(y="survival:Q")
    follow = alt.Chart(pd.DataFrame({"x": [result.max_time]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q")
    st.altair_chart((band + line + follow).properties(height=300), use_container_width=True)
    st.caption(_t("sv_chart_help"))
    st.markdown(f"**{_t('sv_export_title')}**")
    st.caption(_t("sv_export_help"))
    e1, e2, e3 = st.columns(3)
    e1.download_button(_t("sv_dl_table"), table.to_csv(index=False).encode("utf-8"), "nma-survival-curves.csv", "text/csv", key="nma_sv_dl_table")
    n_psa = int(st.number_input(_t("sv_psa_n"), 50, 2000, 500, step=50, key="nma_sv_psa_n"))
    psa = nma_survival.psa_survival_draws(result, np.linspace(0.0, float(horizon), 41), baseline, n_psa)
    e2.download_button(_t("sv_dl_psa"), psa.to_csv(index=False).encode("utf-8"), "nma-survival-psa-draws.csv", "text/csv", key="nma_sv_dl_psa")
    if result.model in ("exponential", "weibull", "gompertz"):
        params = nma_survival.psm_parameter_draws(result, baseline, float(factor), n_psa)
        e3.download_button(_t("sv_dl_psm"), params.to_csv(index=False).encode("utf-8"), "nma-psm-curve-parameters.csv", "text/csv", key="nma_sv_dl_psm")
    else:
        st.caption(_t("sv_psm_note"))



def _read_ipd(upload):
    try:
        return pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(str(exc))
        return None


def _render_population_adjustment() -> None:
    """C2: MAIC and ML-NMR with individual patient data (IPD) that stay in this session."""

    st.caption(_t("pa_intro"))
    st.info(_t("pa_privacy"))
    upload = st.file_uploader(_t("pa_upload"), type=["csv"], key="nma_pa_upload")
    if upload is not None:
        key = (upload.name, upload.size)
        if st.session_state.get("nma_ipd_key") != key:
            frame = _read_ipd(upload)
            if frame is not None:
                st.session_state.nma_ipd, st.session_state.nma_ipd_key = frame, key
    ipd = st.session_state.get("nma_ipd")
    if ipd is None:
        st.caption(_t("pa_need_ipd"))
        return
    st.caption(_t("pa_loaded").format(rows=len(ipd), cols=len(ipd.columns)))
    with st.expander(_t("pa_preview")):
        st.dataframe(ipd.head(20), use_container_width=True)
    maic_tab, ml_tab = st.tabs([_t("pa_maic_tab"), _t("pa_mlnmr_tab")])
    with maic_tab:
        _render_maic(ipd)
    with ml_tab:
        _render_mlnmr(ipd)


def _render_maic(ipd) -> None:
    columns = list(ipd.columns)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(ipd[c])]
    c1, c2 = st.columns(2)
    arm_col = c1.selectbox(_t("pa_arm_col"), columns, key="nma_maic_arm")
    outcome = c2.selectbox(_t("pa_outcome_type"), ["binary", "continuous", "survival"], key="nma_maic_type",
                           format_func=lambda v: _t(f"pa_type_{v}"))
    d1, d2 = st.columns(2)
    values = sorted(ipd[arm_col].dropna().unique().tolist(), key=str)
    if len(values) < 2:
        st.warning(_t("pa_two_arms"))
        return
    treatment = d1.selectbox(_t("pa_treatment"), values, key="nma_maic_trt")
    comparator = d2.selectbox(_t("pa_comparator"), [v for v in values if v != treatment], key="nma_maic_cmp")
    data_cols = {}
    if outcome == "binary":
        data_cols["event"] = st.selectbox(_t("pa_event_col"), numeric, key="nma_maic_event")
    elif outcome == "continuous":
        data_cols["value"] = st.selectbox(_t("pa_value_col"), numeric, key="nma_maic_value")
    else:
        e1, e2 = st.columns(2)
        data_cols["time"] = e1.selectbox(_t("pa_time_col"), numeric, key="nma_maic_time")
        data_cols["event"] = e2.selectbox(_t("pa_event_col"), numeric, key="nma_maic_event_s")
    covariates = st.multiselect(_t("pa_cov_cols"), [c for c in numeric if c not in data_cols.values()], key="nma_maic_covs")
    if not covariates:
        st.info(_t("pa_pick_covs"))
        return
    st.markdown(f"**{_t('pa_target')}**")
    st.caption(_t("pa_target_help"))
    frame = pd.DataFrame({"covariate": covariates, "mean": [float(ipd[c].mean()) for c in covariates], "sd": [None] * len(covariates)})
    target = st.data_editor(frame, hide_index=True, use_container_width=True, disabled=["covariate"],
                            key=f"nma_maic_target_{'|'.join(covariates)}")
    use_sd = st.checkbox(_t("pa_match_sd"), value=False, key="nma_maic_sd")
    method = st.radio(_t("pa_se_method"), ["auto", "bootstrap"], horizontal=True, key="nma_maic_se",
                      format_func=lambda v: _t("pa_se_auto") if v == "auto" else _t("pa_se_boot"))
    if not st.button(_t("pa_run"), type="primary", key="nma_maic_run"):
        stored = st.session_state.get("nma_maic")
        if stored is None:
            return
        effect = stored
    else:
        used = ipd[[arm_col, *covariates, *data_cols.values()]].dropna()
        arm = used[arm_col].to_numpy()
        with st.spinner(_t("pa_running")):
            effect = nma_maic.run_maic(
                used[covariates].to_numpy(float), arm, outcome, {k: used[v].to_numpy(float) for k, v in data_cols.items()},
                target["mean"].to_numpy(float), target["sd"].to_numpy(float) if use_sd else None, covariates,
                treatment=treatment, comparator=comparator, se_method=method)
        st.session_state.nma_maic = effect
    if effect.errors:
        st.error("\n".join(e.message for e in effect.errors))
        return
    for message in effect.warnings:
        st.warning(message)
    w = effect.weights
    st.metric(_t("pa_ess"), f"{effect.ess:.0f} / {effect.n}", f"{100 * effect.ess / effect.n:.0f}%", delta_color="off")
    frames = sup.maic_frames(effect, {"covariate": _t("pa_covariate"), "unweighted": _t("pa_unweighted"), "weighted": _t("pa_weighted"),
                                      "target": _t("pa_target_col"), "analysis": _t("pa_analysis"), "estimate": _t("thr_estimate"),
                                      "unadjusted": _t("pa_unadjusted"), "adjusted": _t("pa_adjusted"), "method": _t("pa_se_method")})
    st.markdown(f"**{_t('pa_balance')}**")
    st.dataframe(frames["balance"], hide_index=True, use_container_width=True)
    st.markdown(f"**{_t('pa_effect')}** ({str(treatment)} vs {str(comparator)}; {_t('pa_scale_' + outcome)})")
    st.dataframe(frames["effects"], hide_index=True, use_container_width=True)
    hist = pd.DataFrame({"weight": w.weights})
    st.altair_chart(alt.Chart(hist).mark_bar().encode(x=alt.X("weight:Q", bin=alt.Bin(maxbins=40), title=_t("pa_weight")),
                                                      y=alt.Y("count()", title=None)).properties(height=140), use_container_width=True)
    st.markdown(f"**{_t('pa_anchor_title')}**")
    st.caption(_t("pa_anchor_help"))
    a1, a2, a3 = st.columns(3)
    agg_est = a1.number_input(_t("pa_agg_effect"), value=1.0 if effect.ratio else 0.0, format="%g", key="nma_maic_agg_est")
    agg_lo = a2.number_input(_t("pa_agg_lower"), value=None, format="%g", key="nma_maic_agg_lo")
    agg_hi = a3.number_input(_t("pa_agg_upper"), value=None, format="%g", key="nma_maic_agg_hi")
    if agg_lo is not None and agg_hi is not None and agg_lo < agg_hi:
        tr = math.log if effect.ratio else (lambda v: v)
        if effect.ratio and min(agg_est, agg_lo, agg_hi) <= 0:
            st.warning(_t("pa_agg_positive"))
        else:
            se_agg = (tr(agg_hi) - tr(agg_lo)) / (2 * 1.959963984540054)
            est, se = nma_maic.anchored_indirect(effect.estimate, effect.se, tr(agg_est), se_agg)
            nat = math.exp if effect.ratio else (lambda v: v)
            st.success(_t("pa_anchor_result").format(
                est=f"{nat(est):.3f}", lo=f"{nat(est - 1.959963984540054 * se):.3f}", hi=f"{nat(est + 1.959963984540054 * se):.3f}"))
    st.session_state.nma_popadj_maic = frames


def _render_mlnmr(ipd) -> None:
    document = st.session_state.nma_document
    columns = list(ipd.columns)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(ipd[c])]
    st.caption(_t("pa_ml_intro"))
    c1, c2, c3 = st.columns(3)
    study_col = c1.selectbox(_t("pa_study_col"), columns, key="nma_ml_study")
    arm_col = c2.selectbox(_t("pa_arm_col"), columns, index=min(1, len(columns) - 1), key="nma_ml_arm")
    event_col = c3.selectbox(_t("pa_event_col"), numeric, key="nma_ml_event")
    covs = st.multiselect(_t("pa_cov_cols"), [c for c in numeric if c != event_col], key="nma_ml_covs")
    if not covs:
        st.info(_t("pa_pick_covs"))
        return
    binary = [bool(set(ipd[c].dropna().unique()) <= {0, 1}) for c in covs]
    mods = st.multiselect(_t("pa_em_cols"), covs, default=covs[:1], key="nma_ml_em")
    treatments = sorted({str(v) for v in ipd[arm_col].dropna().unique()})
    agg_saved = document.get("mlnmr_agg") or []
    base_cols = ["study", "treatment", "events", "n"] + [f"{p}_{c}" for c in covs for p in ("mean", "sd")]
    st.markdown(f"**{_t('pa_agg_title')}**")
    st.caption(_t("pa_agg_help"))
    agg_frame = pd.DataFrame(agg_saved, columns=base_cols) if agg_saved else pd.DataFrame(
        {c: pd.Series(dtype="object" if c in ("study", "treatment") else "float64") for c in base_cols})
    edited = st.data_editor(agg_frame, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"nma_ml_agg_{'|'.join(covs)}")
    rows = json.loads(edited.to_json(orient="records"))
    if rows != agg_saved:
        document["mlnmr_agg"] = rows
    d1, d2 = st.columns(2)
    all_treatments = sorted(set(treatments) | {str(r["treatment"]) for r in rows if r.get("treatment")})
    reference = d1.selectbox(_t("tv_reference"), all_treatments, key="nma_ml_ref")
    shared = d2.checkbox(_t("pa_shared"), value=True, key="nma_ml_shared", help=_t("pa_shared_help"))
    st.markdown(f"**{_t('pa_profile')}**")
    profile_frame = pd.DataFrame({"covariate": covs, "value": [float(ipd[c].mean()) for c in covs]})
    profile = st.data_editor(profile_frame, hide_index=True, use_container_width=True, disabled=["covariate"],
                             key=f"nma_ml_profile_{'|'.join(covs)}")
    if not st.button(_t("pa_run"), type="primary", key="nma_ml_run"):
        result = st.session_state.get("nma_mlnmr")
        if result is None:
            return
    else:
        used = ipd[[study_col, arm_col, event_col, *covs]].dropna()
        ipd_arms = [nma_mlnmr.IPDArm(str(s), str(a), g[covs].to_numpy(float), g[event_col].to_numpy(float))
                    for (s, a), g in used.groupby([study_col, arm_col])]
        agg_arms = []
        for r in rows:
            try:
                if r.get("study") and r.get("treatment") and r.get("events") is not None and r.get("n"):
                    agg_arms.append(nma_mlnmr.AggArm(
                        str(r["study"]), str(r["treatment"]), float(r["events"]), float(r["n"]),
                        np.array([float(r[f"mean_{c}"]) for c in covs]),
                        np.array([float(r[f"sd_{c}"]) if r.get(f"sd_{c}") is not None else 1.0 for c in covs])))
            except (TypeError, ValueError, KeyError):
                continue
        with st.spinner(_t("pa_running")):
            result = nma_mlnmr.run_mlnmr(ipd_arms, agg_arms, covs, binary, mods, str(reference), shared=shared)
        st.session_state.nma_mlnmr = result
    if result.errors:
        st.error("\n".join(e.message for e in result.errors))
        return
    for message in result.warnings:
        st.warning(message)
    frames = sup.mlnmr_frames(result, profile["value"].to_numpy(float), {
        "parameter": _t("pa_parameter"), "estimate": _t("thr_estimate"), "treatment": _t("thr_col_treatment"),
        "or": _t("pa_or"), "profile": _t("pa_profile")})
    st.markdown(f"**{_t('pa_ml_effects')}**")
    st.dataframe(frames["effects"], hide_index=True, use_container_width=True)
    st.markdown(f"**{_t('pa_coefficients')}**")
    st.dataframe(frames["coefficients"], hide_index=True, use_container_width=True)
    st.caption(_t("pa_ml_note"))
    st.download_button(_t("thr_download"), frames["effects"].to_csv(index=False).encode("utf-8"), "nma-mlnmr-effects.csv", "text/csv", key="nma_ml_dl")
    st.session_state.nma_popadj_mlnmr = sup.mlnmr_frames(result, profile["value"].to_numpy(float))



DOSE_COLUMNS = ["study", "agent", "dose", "mean", "se", "events", "n"]
DOSE_FUNCTIONS = {"linear": "Linear", "loglinear": "Log-linear", "quadratic": "Quadratic", "emax": "Emax",
                  "emax_hill": "Emax with Hill", "exp_sat": "Exponential (saturating)"}


def _render_dose_response() -> None:
    """C3: dose-response MBNMA (maximum likelihood)."""

    document = st.session_state.nma_document
    st.caption(_t("dr_intro"))
    rows = _simple_table_editor(document, "dose_arms", DOSE_COLUMNS, "nma_dr_arms")
    arms, kind = [], None
    for r in rows:
        try:
            if not (r.get("study") and r.get("agent")) or r.get("dose") is None:
                continue
            if r.get("events") is not None and r.get("n"):
                arms.append(nma_mbnma.DoseArm(str(r["study"]), str(r["agent"]), float(r["dose"]), events=float(r["events"]), n=float(r["n"])))
                kind = "binary"
            elif r.get("mean") is not None and r.get("se"):
                arms.append(nma_mbnma.DoseArm(str(r["study"]), str(r["agent"]), float(r["dose"]), mean=float(r["mean"]), se=float(r["se"])))
                kind = "continuous"
        except (TypeError, ValueError):
            continue
    if len(arms) < 4:
        st.info(_t("dr_none"))
        return
    agents = sorted({a.agent for a in arms if a.dose > 0})
    saved_groups = document.get("dose_groups") or {}
    with st.expander(_t("dr_groups_title")):
        st.caption(_t("dr_groups_help"))
        gframe = pd.DataFrame({"agent": agents, "group": [saved_groups.get(a, a) for a in agents]})
        gedit = st.data_editor(gframe, hide_index=True, use_container_width=True, disabled=["agent"], key=f"nma_dr_groups_{'|'.join(agents)}")
        groups = {str(r["agent"]): str(r["group"] or r["agent"]) for _, r in gedit.iterrows()}
        if groups != saved_groups:
            document["dose_groups"] = groups
    mixed = {a.events is not None for a in arms}
    if len(mixed) > 1:
        st.warning(_t("dr_mixed"))
        return
    chosen = st.multiselect(_t("dr_functions"), list(DOSE_FUNCTIONS), default=["linear", "emax"], key="nma_dr_functions",
                            format_func=lambda k: DOSE_FUNCTIONS[k])
    stored = st.session_state.setdefault("nma_dose", {"fits": {}, "selected": None, "key": None})
    key = json.dumps([rows, groups], sort_keys=True, default=str)
    if stored["key"] != key:
        stored.update({"fits": {}, "selected": None, "key": key})
    if st.button(_t("sv_run"), type="primary", key="nma_dr_run"):
        for name in chosen:
            fit = nma_mbnma.run_mbnma(arms, name, groups)
            if fit.errors:
                st.error(f"{DOSE_FUNCTIONS[name]}: " + "; ".join(e.message for e in fit.errors))
                continue
            stored["fits"][DOSE_FUNCTIONS[name]] = fit
            stored["selected"] = DOSE_FUNCTIONS[name]
    fits = stored["fits"]
    if not fits:
        return
    st.markdown(f"**{_t('dr_compare')}**")
    st.dataframe(sup.mbnma_models_frame(fits, {"function": _t("dr_function"), "dev": _t("dr_deviance"), "p": _t("dr_fit_p"),
                                               "disp": _t("dr_dispersion")}), hide_index=True, use_container_width=True)
    labels = list(fits)
    selected = st.selectbox(_t("sv_show"), labels, index=labels.index(stored["selected"]) if stored["selected"] in labels else 0, key="nma_dr_show")
    stored["selected"] = selected
    fit = fits[selected]
    for message in fit.warnings:
        st.warning(message)
    st.dataframe(sup.mbnma_params_frame(fit, {"parameter": _t("pa_parameter"), "estimate": _t("thr_estimate")}), hide_index=True, use_container_width=True)
    st.caption(_t("dr_param_help"))
    top = max(fit.max_dose.values())
    curves, follow = [], []
    for agent in agents:
        grid = np.linspace(0, top * 1.3, 60)
        f, lo, hi = fit.curve(agent, grid)
        curves.append(pd.DataFrame({"dose": grid, "effect": f, "lower": lo, "upper": hi, "agent": agent}))
        follow.append({"agent": agent, "x": fit.max_dose[agent]})
    table = pd.concat(curves)
    base = alt.Chart(table).encode(x=alt.X("dose:Q", title=_t("dr_dose")), color=alt.Color("agent:N", title=None))
    band = base.mark_area(opacity=0.15).encode(y=alt.Y("lower:Q", title=_t("dr_effect_axis_" + fit.outcome)), y2="upper:Q")
    line = base.mark_line().encode(y="effect:Q")
    rules = alt.Chart(pd.DataFrame(follow)).mark_rule(strokeDash=[4, 4]).encode(x="x:Q", color="agent:N")
    layers = band + line + rules
    obs = []
    by_study = {}
    for a in arms:
        by_study.setdefault(a.study, []).append(a)
    for s, items in by_study.items():
        zero = [a for a in items if a.dose == 0]
        if not zero:
            continue
        for a in items:
            if a.dose > 0:
                if fit.outcome == "binary":
                    lg = lambda ev, nn: math.log(min(max(ev / nn, 0.005), 0.995) / (1 - min(max(ev / nn, 0.005), 0.995)))  # noqa: E731
                    obs.append({"dose": a.dose, "effect": lg(a.events, a.n) - lg(zero[0].events, zero[0].n), "agent": a.agent, "study": s})
                else:
                    obs.append({"dose": a.dose, "effect": a.mean - zero[0].mean, "agent": a.agent, "study": s})
    if obs:
        layers = layers + alt.Chart(pd.DataFrame(obs)).mark_point(filled=True, size=45, opacity=0.7).encode(
            x="dose:Q", y="effect:Q", color="agent:N", tooltip=["study", "agent", "dose", "effect"])
    st.altair_chart(layers.properties(height=320), use_container_width=True)
    st.caption(_t("dr_chart_help"))
    st.download_button(_t("thr_download"), table.to_csv(index=False).encode("utf-8"), "nma-dose-response-curves.csv", "text/csv", key="nma_dr_dl")



# --------------------------------------------------------------------------
# Data conversion tools, ranking across outcomes, prior sensitivity
# --------------------------------------------------------------------------


def _optional(label: str, key: str):
    """Number input that may be left empty (returns None)."""

    return st.number_input(label, value=None, format="%g", key=key)


def _render_conversions(spec: dict, treatments_df=None, arms_df=None, contrasts_df=None, chars_df=None) -> None:
    fill = {}  # what the current tool produced, for the 'fill into table' button
    tools = {"median": _t("conv_median"), "se": _t("conv_se"), "sd": _t("conv_sd"), "combine": _t("conv_combine")}
    tool = st.selectbox(_t("conv_tool"), list(tools), format_func=lambda key: tools[key], key="nma_conv_tool")
    try:
        if tool == "median":
            cols = st.columns(3)
            with cols[0]:
                n = _optional(_t("conv_n"), "nma_conv_n")
                median = _optional(_t("conv_median_v"), "nma_conv_median")
            with cols[1]:
                q1 = _optional("Q1", "nma_conv_q1")
                q3 = _optional("Q3", "nma_conv_q3")
            with cols[2]:
                minimum = _optional("Min", "nma_conv_min")
                maximum = _optional("Max", "nma_conv_max")
            if n is not None and median is not None:
                result = mean_sd_from_median(n, median, minimum, maximum, q1, q3)
                st.success(f"{_t('conv_mean')} = **{result.mean:.4g}** · SD = **{result.sd:.4g}**")
                st.caption(f"{result.method}. {_t('conv_note_median')}")
                fill = {"table": "arms", "mean": result.mean, "sd": result.sd, "n": n}
        elif tool == "se":
            ratio_default = is_ratio_measure(spec.get("measure", "OR"))
            ratio = st.toggle(_t("conv_ratio"), value=ratio_default, key="nma_conv_ratio")
            source = st.segmented_control(_t("conv_from"), ["ci", "p"], default="ci",
                                          format_func=lambda key: _t("conv_ci") if key == "ci" else _t("conv_p"),
                                          key="nma_conv_source") or "ci"
            cols = st.columns(3)
            with cols[0]:
                estimate = _optional(_t("conv_estimate"), "nma_conv_est")
            if source == "ci":
                with cols[1]:
                    lower = _optional(_t("conv_lower"), "nma_conv_lower")
                with cols[2]:
                    upper = _optional(_t("conv_upper"), "nma_conv_upper")
                if lower is not None and upper is not None:
                    se = se_from_ci(lower, upper, ratio, float(spec.get("ci_level", 0.95)))
                    fill = _show_se(estimate, se, ratio)
            else:
                with cols[1]:
                    p_value = _optional(_t("conv_p"), "nma_conv_p")
                if estimate is not None and p_value is not None:
                    fill = _show_se(estimate, se_from_p(estimate, p_value, ratio), ratio)
        elif tool == "sd":
            source = st.segmented_control(_t("conv_from"), ["se", "ci"], default="se",
                                          format_func=lambda key: "SE" if key == "se" else _t("conv_ci"),
                                          key="nma_conv_sd_source") or "se"
            cols = st.columns(3)
            with cols[0]:
                n = _optional(_t("conv_n"), "nma_conv_sd_n")
            if source == "se":
                with cols[1]:
                    se = _optional("SE", "nma_conv_sd_se")
                if n is not None and se is not None:
                    st.success(f"SD = **{sd_from_se(se, n):.4g}**")
                    fill = {"table": "arms", "sd": sd_from_se(se, n), "n": n}
            else:
                with cols[1]:
                    lower = _optional(_t("conv_lower"), "nma_conv_sd_lower")
                with cols[2]:
                    upper = _optional(_t("conv_upper"), "nma_conv_sd_upper")
                if n is not None and lower is not None and upper is not None:
                    sd = sd_from_ci(lower, upper, n, float(spec.get("ci_level", 0.95)))
                    st.success(f"SD = **{sd:.4g}**")
                    fill = {"table": "arms", "sd": sd, "n": n}
        else:
            arm_cols = st.columns(2)
            values = []
            for i, col in enumerate(arm_cols, start=1):
                with col:
                    st.markdown(f"**{_t('conv_arm')} {i}**")
                    values.append((_optional(_t("conv_n"), f"nma_conv_c_n{i}"),
                                   _optional(_t("conv_mean"), f"nma_conv_c_m{i}"),
                                   _optional("SD", f"nma_conv_c_sd{i}")))
            if all(v is not None for group in values for v in group):
                n, mean, sd = combine_arms(*values[0], *values[1])
                st.success(f"n = **{n:g}** · {_t('conv_mean')} = **{mean:.4g}** · SD = **{sd:.4g}**")
                fill = {"table": "arms", "mean": mean, "sd": sd, "n": n}
            st.caption(_t("conv_combine_binary"))
    except ValueError as exc:
        st.warning(str(exc))
        return
    if fill and arms_df is not None and contrasts_df is not None:
        _render_fill_into_table(tool, fill, treatments_df, arms_df, contrasts_df, chars_df)


def _is_blank(value) -> bool:
    if value is None or value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _render_fill_into_table(tool: str, fill: dict, treatments_df, arms_df, contrasts_df, chars_df) -> None:
    """Write the converted numbers into a chosen row of the arm-level or
    contrast-level table (or a new row), then refresh the editor."""

    table = fill.pop("table")
    frame = arms_df if table == "arms" else contrasts_df
    values = {key: value for key, value in fill.items() if value is not None}
    if not values:
        return
    st.markdown(f"**{_t('conv_fill_title')}** · {_t('arm_level') if table == 'arms' else _t('contrast_level')}")
    labels: list[str] = []
    for index, row in frame.iterrows():
        study = "" if _is_blank(row.get("study")) else str(row.get("study"))
        treatment = "" if _is_blank(row.get("treatment")) else str(row.get("treatment"))
        if table == "contrasts":
            versus = "" if _is_blank(row.get("versus")) else str(row.get("versus"))
            treatment = f"{treatment} vs {versus}" if versus else treatment
        labels.append(f"{index + 1}. {study or '—'} · {treatment or '—'}")
    options = {-1: _t("conv_fill_new"), **{i: label for i, label in enumerate(labels)}}
    target_col, button_col = st.columns([3, 1.4], vertical_alignment="bottom")
    with target_col:
        position = st.selectbox(_t("conv_fill_target"), list(options), format_func=options.get,
                                key=f"nma_conv_fill_target_{tool}")
    position = -1 if position is None else int(position)
    existing = {}
    if position >= 0:
        existing = {key: frame.iloc[position][key] for key in values
                    if key in frame.columns and not _is_blank(frame.iloc[position][key])}
    overwrite = True
    if existing:
        st.warning(_t("conv_fill_existing").format(values=", ".join(f"{k} = {v}" for k, v in existing.items())))
        overwrite = st.checkbox(_t("conv_fill_overwrite"), value=False, key=f"nma_conv_fill_overwrite_{tool}")
    with button_col:
        clicked = st.button(_t("conv_fill_button"), icon=":material/input:", disabled=not overwrite,
                            key=f"nma_conv_fill_button_{tool}", width="stretch")
    if not clicked or not overwrite:
        return
    updated = frame.copy().astype(object)
    if position < 0:
        updated = pd.concat([updated, pd.DataFrame([{column: None for column in updated.columns}])], ignore_index=True)
        position = len(updated) - 1
    for key, value in values.items():
        if key in updated.columns:
            updated.at[position, key] = float(value)
    if table == "arms":
        _commit_tables(treatments_df, updated, contrasts_df, chars_df)
    else:
        _commit_tables(treatments_df, arms_df, updated, chars_df)
    st.session_state["nma_conv_fill_message"] = _t("conv_fill_done").format(
        what=", ".join(f"{k} = {float(v):.4g}" for k, v in values.items()), row=position + 1)
    st.rerun()


def _show_se(estimate, se: float, ratio: bool) -> dict:
    text = f"SE = **{se:.4g}**" + (" (log)" if ratio else "")
    fill = {"table": "contrasts", "se": se}
    if estimate is not None and (not ratio or estimate > 0):
        value = math.log(estimate) if ratio else estimate
        text += f" · {_t('conv_estimate')} ({'log' if ratio else ''}) = **{value:.4g}**"
        fill["estimate"] = value
    st.success(text)
    st.caption(_t("conv_copy"))
    return fill


def _analysis_ranking(document: dict) -> tuple[dict[str, float] | None, str]:
    """P-scores of one analysis (cached by its data), or (None, error)."""

    cache = st.session_state.setdefault("nma_rank_cache", {})
    key = (sup.document_hash(document), document.get("spec", {}).get("effects", "random"))
    if key not in cache:
        try:
            model = resolve_nma(document)
            errors = validate_nma(model)
            if errors:
                cache[key] = (None, errors[0].message)
            else:
                result = run_nma_frequentist(model)
                fit = result.random if key[1] == "random" else result.fixed
                cache[key] = (dict(fit.p_scores), "") if not result.errors else (None, result.errors[0].message)
        except ValueError as exc:
            cache[key] = (None, str(exc))
        if len(cache) > 32:
            cache.pop(next(iter(cache)))
    return cache[key]


def _render_rank_heat() -> None:
    documents = _all_documents()
    if len(documents) < 2:
        st.info(_t("rank_single"))
        return
    st.caption(_t("rank_intro"))
    rankings = []
    for index, document in enumerate(documents):
        title = sup.analysis_title(document, index)
        scores, error = _analysis_ranking(document)
        if scores is None:
            st.caption(_t("rank_failed").format(title=title, error=error))
        else:
            rankings.append((title, scores))
    if not rankings:
        return
    order = [t["name"] for t in st.session_state.nma_document.get("treatments", []) if t.get("name")]
    frame = sup.rank_heat_frame(rankings, order)
    st.markdown(sup.rank_heat_html(frame, _t("rank_treatment")), unsafe_allow_html=True)
    st.download_button(_t("download_csv"), (frame * 100).round(1).to_csv().encode("utf-8"),
                       "nma-ranking-across-outcomes.csv", "text/csv", key="nma_rank_download")
    if len(rankings) >= 2:
        titles = list(frame.columns)
        x_col, y_col = st.columns(2)
        with x_col:
            x = st.selectbox(_t("rank_x"), titles, index=0, key="nma_rank_x")
        with y_col:
            y = st.selectbox(_t("rank_y"), titles, index=1, key="nma_rank_y")
        points = frame[[x, y]].dropna().reset_index()
        points.columns = ["treatment", "x", "y"]
        if not points.empty:
            base = alt.Chart(points).encode(
                x=alt.X("x:Q", title=f"{x} (P-score)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", title=f"{y} (P-score)", scale=alt.Scale(domain=[0, 1])))
            dots = base.mark_circle(size=140, color="#1e3a8a", opacity=0.85).encode(
                tooltip=["treatment", alt.Tooltip("x:Q", title=x, format=".2f"),
                         alt.Tooltip("y:Q", title=y, format=".2f")])
            labels = base.mark_text(dy=-12, fontSize=12, fontWeight=600, color="#0f172a").encode(text="treatment:N")
            guides = alt.Chart(pd.DataFrame({"v": [0.5]}))
            st.altair_chart(alt.layer(guides.mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(x="v:Q"),
                                      guides.mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(y="v:Q"),
                                      dots, labels).properties(height=380), use_container_width=True)
            st.caption(_t("rank_scatter"))


PRIOR_SET = [("uniform", 2.0, "Uniform(0, 2)"), ("uniform", 5.0, "Uniform(0, 5)"),
             ("half-normal", 0.5, "Half-normal(0, 0.5)"), ("half-normal", 1.0, "Half-normal(0, 1)")]


def _render_prior_sensitivity() -> None:
    spec = st.session_state.nma_document.get("spec", {})
    st.markdown(f"**{_t('prior_title')}**")
    if spec.get("engine") != "bayesian":
        st.caption(_t("prior_needs"))
        return
    st.caption(_t("prior_help"))
    key = sup.bayes_hash(st.session_state.nma_document)
    stored = st.session_state.get("nma_prior_sensitivity")
    if st.button(_t("prior_run"), icon=":material/tune:", key="nma_prior_run"):
        try:
            model = resolve_nma(st.session_state.nma_document)
        except ValueError as exc:
            st.error(str(exc))
            return
        bayes = spec.get("bayes", {})
        runs = []
        exact = _uses_exact(spec)
        prior_set = list(PRIOR_SET)
        informative = _informative_tau(spec)
        if informative is not None:
            prior_set.append(("lognormal", None, f"{_t('tau_informative')} – {informative['label']}"))
        with st.spinner(_t("prior_run")):
            for prior, value, label in prior_set:
                if prior == "lognormal":
                    settings = _bayes_settings(spec, "random", tau_prior="lognormal", tau_mu=informative["mu"],
                                               tau_sigma=informative["sigma"], tau_df=informative["df"],
                                               tau_prior_label=informative["label"])
                    current = bayes.get("tau_prior") == "lognormal"
                else:
                    settings = _bayes_settings(spec, "random", tau_prior=prior,
                                               tau_max=value if prior == "uniform" else 5.0,
                                               tau_scale=value if prior == "half-normal" else 0.5)
                    current = (prior == bayes.get("tau_prior", "uniform")
                               and value == (float(bayes.get("tau_max", 5.0)) if prior == "uniform"
                                             else float(bayes.get("tau_scale", 0.5))))
                run = (run_nma_bayesian_exact(model, settings, warmup=int(bayes.get("warmup", 1000))) if exact
                       else run_nma_bayesian(model, settings))
                runs.append((label + (" " + _t("prior_current") if current else ""), run))
        stored = {"key": key, "runs": runs}
        st.session_state.nma_prior_sensitivity = stored
    if not stored:
        return
    if stored["key"] != key:
        st.warning(_t("stale"))
    rows = []
    for label, run in stored["runs"]:
        if run.errors:
            rows.append({_t("prior_col"): label, "τ": "; ".join(e.message for e in run.errors)})
            continue
        row = {_t("prior_col"): label,
               "τ (CrI)": f"{run.tau.median:.3f} ({run.tau.lower:.3f}–{run.tau.upper:.3f})" if run.tau else "–"}
        for effect in run.fit.effects:
            row[f"{effect.treatment} vs {effect.versus}"] = sup.format_effect(
                effect.natural, effect.natural_lower, effect.natural_upper)
        row["DIC"] = round(run.dic, 1) if run.dic is not None else None
        rows.append(row)
    _show_table(pd.DataFrame(rows))


def _scale_label(measure: str) -> str:
    return f"log {measure}" if is_ratio_measure(measure) else measure


def _render_meta_regression(model) -> None:
    document = st.session_state.nma_document
    characteristics = document.get("study_characteristics", [])
    options: list[str] = []
    for key in sup.numeric_characteristic_columns(document):
        count = sum(1 for row in characteristics if isinstance(row.get(key), (int, float)))
        if count >= 3:
            options.append(key)
    if model.spec.outcome in ("binary", "rate", "continuous") and model.spec.measure != "SMD":
        options.append(BASELINE_RISK)
    if not options:
        st.info(_t("nmr_no_covariate"))
        return
    labels = {**sup.characteristic_labels(document), BASELINE_RISK: _t("nmr_baseline")}
    cov_col, int_col = st.columns([1, 1.3])
    with cov_col:
        covariate = st.selectbox(_t("nmr_covariate"), options, format_func=lambda key: labels[key], key="nma_nmr_covariate")
    with int_col:
        interaction_labels = {"shared": _t("nmr_shared"), "specific": _t("nmr_specific")}
        interaction = st.segmented_control(_t("nmr_interaction"), ["shared", "specific"], default="shared",
                                           format_func=lambda key: interaction_labels[key],
                                           key="nma_nmr_interaction") or "shared"
    effects = document.get("spec", {}).get("effects", "random")
    values = {str(row["study"]): float(row[covariate]) for row in characteristics
              if covariate != BASELINE_RISK and isinstance(row.get(covariate), (int, float))}
    result = run_meta_regression(model, covariate, values, interaction, effects)
    if result.errors:
        st.warning("\n".join(error.message for error in result.errors))
        return

    scale = _scale_label(result.measure)
    st.markdown(f"**{_t('nmr_coefficients').format(scale=scale)}** · {effects} effects")
    _show_table(pd.DataFrame([{
        "β for": ("all vs " + result.reference) if coefficient.treatment == "all" else f"{coefficient.treatment} vs {result.reference}",
        "Estimate": round(coefficient.estimate, 4), "SE": round(coefficient.se, 4),
        "CI lower": round(coefficient.lower, 4), "CI upper": round(coefficient.upper, 4),
        "p": _fmt_p(coefficient.p_value),
    } for coefficient in result.coefficients]), flags=[c.p_value < 0.05 for c in result.coefficients])
    if effects == "random":
        explained = result.tau2_explained
        st.caption(_t("nmr_tau").format(before=result.tau2_without, after=result.tau2,
                                        explained="" if explained is None else f" ({explained:.0%} explained)"))

    low, high = result.covariate_range
    at = st.number_input(_t("nmr_effects_at").format(ref=result.reference) + labels[covariate],
                         value=float(round(result.centre, 3)), key=f"nma_nmr_at_{covariate}")
    adjusted = effects_at(result, at, model.spec.ci_level)
    _show_table(pd.DataFrame([{
        "Comparison": f"{effect.treatment} vs {effect.versus}",
        f"{result.measure} (CI)": sup.format_effect(effect.natural, effect.natural_lower, effect.natural_upper),
    } for effect in adjusted]))
    if not (low <= at <= high):
        st.caption(f"⚠ {at:g} is outside the observed range {low:g}–{high:g} (extrapolation).")

    treatment = st.selectbox(_t("nmr_curve_treatment"), [e.treatment for e in adjusted], key="nma_nmr_curve_treatment")
    curve = effect_curve(result, treatment, ci_level=model.spec.ci_level)
    if curve:
        st.markdown(f"**{_t('nmr_curve').format(ref=result.reference)}** · {treatment}")
        frame = pd.DataFrame([{"x": e.at, "estimate": e.natural, "lower": e.natural_lower, "upper": e.natural_upper}
                              for e in curve])
        ratio = is_ratio_measure(result.measure)
        y_scale = alt.Scale(type="log") if ratio else alt.Scale(zero=False)
        base = alt.Chart(frame).encode(x=alt.X("x:Q", title=labels[covariate]))
        band = base.mark_area(opacity=0.25, color="#2f6fd6").encode(
            y=alt.Y("lower:Q", scale=y_scale, title=f"{result.measure} vs {result.reference}"), y2="upper:Q")
        line = base.mark_line(color="#1e3a8a", strokeWidth=2.5).encode(y=alt.Y("estimate:Q", scale=y_scale))
        null = alt.Chart(pd.DataFrame({"y": [1.0 if ratio else 0.0]})).mark_rule(
            strokeDash=[4, 4], color="#334155").encode(y=alt.Y("y:Q", scale=y_scale))
        rug = alt.Chart(pd.DataFrame({"x": list(result.study_values.values())})).mark_tick(
            color="#ea580c", thickness=2, size=12).encode(x="x:Q")
        st.altair_chart(alt.layer(band, null, line, rug).properties(height=260), use_container_width=True)
    if result.excluded_studies:
        st.caption(_t("nmr_excluded").format(ids=", ".join(result.excluded_studies)))
    _show_warnings(result.warnings)
    st.caption(_t("nmr_caveat"))


def _render_funnel(model) -> None:
    order = list(model.treatment_names)
    if st.checkbox(_t("funnel_reverse"), value=False, key="nma_funnel_reverse"):
        order = order[::-1]
    st.caption(_t("funnel_order").format(order=" → ".join(order)))
    funnel = _cached("funnel", tuple(order), lambda: comparison_adjusted_funnel(model, order))
    if funnel.errors:
        st.warning("\n".join(error.message for error in funnel.errors))
        return
    if not funnel.points:
        st.info("No study contrasts to plot.")
        return
    scale = _scale_label(funnel.measure)
    frame = pd.DataFrame([{
        "centred": point.centred, "se": point.se, "study": point.study_id,
        "comparison": f"{point.newer} vs {point.older}",
    } for point in funnel.points])
    max_se = float(frame["se"].max()) * 1.05
    z = 1.959963984540054
    limits = pd.DataFrame({"x": [-z * max_se, 0.0, z * max_se], "se": [max_se, 0.0, max_se]})
    y_axis = alt.Y("se:Q", title="Standard error", scale=alt.Scale(reverse=True, domain=[0, max_se]))
    cone = alt.Chart(limits).mark_line(color="#94a3b8", strokeDash=[5, 4]).encode(
        x=alt.X("x:Q", title=_t("funnel_x").format(scale=scale)), y=y_axis, order="se:Q")
    centre = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#334155").encode(x="x:Q")
    points = alt.Chart(frame).mark_circle(size=80, opacity=0.85, stroke="#ffffff", strokeWidth=1).encode(
        x="centred:Q", y=y_axis,
        color=alt.Color("comparison:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["study", "comparison", alt.Tooltip("centred:Q", format=".3f"), alt.Tooltip("se:Q", format=".3f")])
    st.altair_chart(alt.layer(cone, centre, points).properties(height=360), use_container_width=True)
    if funnel.egger_intercept is not None:
        st.markdown(_t("funnel_egger").format(a=funnel.egger_intercept, se=funnel.egger_se or 0.0,
                                              p=_fmt_p(funnel.egger_p)))
        if funnel.egger_p is not None:
            (st.warning if funnel.egger_p < 0.10 else st.success)(
                _t("funnel_asym") if funnel.egger_p < 0.10 else _t("funnel_sym"))
    _show_warnings(funnel.warnings)


def _render_check_body(summary, model) -> None:
    st.markdown(f"**{_t('check_title')}**")
    edge_rows = [{
        "comparison": f"{a} vs {b}", "studies": edge.n_studies,
        "patients": int(edge.patients), "study ids": ", ".join(edge.study_ids),
    } for (a, b), edge in summary.edges.items()]
    _show_table(pd.DataFrame(edge_rows))
    loops = [" – ".join(loop.treatments) + (" (single multi-arm study)" if loop.single_study else "")
             for loop in summary.loops]
    st.markdown(f"Closed loops: {'; '.join(loops) if loops else _t('none')}")
    st.markdown(f"Multi-arm studies: {', '.join(summary.multi_arm_studies) or _t('none')}")
    st.markdown(f"Components: {len(summary.components)}")
    st.markdown(f"**{_t('transitivity')}**")
    _show_table(sup.transitivity_frame(summary, st.session_state.nma_document.get("study_characteristics", []),
                                       sup.characteristic_variables(st.session_state.nma_document)))
    st.caption(_t("transitivity_hint"))
    _render_transitivity_plot(summary)


ROB_PLOT_COLORS = {"Low": "#16a34a", "Some concerns": "#f59e0b", "High": "#dc2626", "Not recorded": "#94a3b8"}


def _render_transitivity_plot(summary) -> None:
    """Dot plot of each numeric study characteristic by comparison (dots =
    studies coloured by risk of bias, box = interquartile range), plus counts for
    text characteristics."""

    document = st.session_state.nma_document
    variables = sup.characteristic_variables(document)
    rows = document.get("study_characteristics", [])
    points = sup.transitivity_points(summary, rows, variables)
    st.markdown(f"**{_t('transitivity_plot')}**")
    if points.empty:
        st.caption(_t("transitivity_plot_empty"))
    else:
        labels = list(dict.fromkeys(points["label"]))
        choice = st.selectbox(_t("transitivity_variable"), labels, key="nma_transitivity_variable")
        sub = points[points["label"] == choice]
        order = [f"{a} vs {b}" for (a, b) in summary.edges]
        base = alt.Chart(sub).encode(y=alt.Y("comparison:N", sort=order, title=None))
        x = alt.X("value:Q", title=choice, scale=alt.Scale(zero=False))
        box = base.mark_boxplot(extent="min-max", size=16, opacity=0.4, color="#93c5fd").encode(x=x)
        dots = base.mark_circle(size=120, opacity=0.95, stroke="white", strokeWidth=1).encode(
            x=x, color=alt.Color("rob:N", title=None, legend=alt.Legend(orient="bottom"),
                                 scale=alt.Scale(domain=list(ROB_PLOT_COLORS), range=list(ROB_PLOT_COLORS.values()))),
            tooltip=[alt.Tooltip("study:N"), alt.Tooltip("value:Q"), alt.Tooltip("rob:N", title="Risk of bias")])
        st.altair_chart((box + dots).properties(height=max(120, 48 * len(order))), use_container_width=True)
        st.caption(_t("transitivity_plot_hint"))
    counts = sup.transitivity_categories(summary, rows, variables)
    if not counts.empty:
        st.markdown(f"**{_t('transitivity_text_counts')}**")
        _show_table(counts.rename(columns={"variable": "Variable", "comparison": "Comparison",
                                           "category": "Category", "studies": "Studies"}))


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def render_nma_editor(T: dict[str, str], ctx: dict[str, Any]) -> None:
    """ctx: LANGUAGES, render_collapsible_panel."""

    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    _offer_autosave_restore()
    document = st.session_state.nma_document
    spec = document.setdefault("spec", {})
    version = st.session_state.nma_editor_version

    # ---- Toolbar ---------------------------------------------------------
    with st.container(key="nma_toolbar_sticky"):
        (logo_col, title_col, spacer_col, back_col, check_col, run_col,
         incons_col, dash_col, more_col, ee_col) = st.columns(
            [0.9, 1.5, 0.1, 0.9, 1.0, 0.8, 0.95, 0.9, 1.05, 1.0], vertical_alignment="center", gap="small")
        with logo_col:
            if os.environ.get("DECIDE_DESKTOP") != "1":
                st.image("assets/decide-logo-transparent.png", width=130, output_format="PNG")
        with title_col:
            st.markdown(f"### {_t('title')}")
            st.caption(_t("subtitle"))
        with back_col:
            if st.button(T["change_model"], icon=":material/arrow_back:", key="nma_back_to_picker", use_container_width=True):
                st.session_state.model_type = None
                st.rerun()
        with check_col:
            if st.button(_t("check_network"), icon=":material/fact_check:", key="nma_check_network", width="stretch"):
                st.session_state.nma_check_open = not st.session_state.nma_check_open
        with run_col:
            if st.button(_t("run_nma"), icon=":material/play_arrow:", type="primary", key="nma_run_model", width="stretch"):
                st.session_state.nma_run_requested = True
                st.session_state.nma_results_open = True
                st.session_state.nma_results_collapsed = False
                st.session_state.panel_scroll_target = "nma_results_panel_anchor"
        with incons_col:
            if st.button(_t("inconsistency"), icon=":material/call_split:", help=_t("inconsistency_help"),
                         key="nma_open_inconsistency", width="stretch"):
                st.session_state.nma_incons_open = not st.session_state.nma_incons_open
                if st.session_state.nma_incons_open:
                    st.session_state.nma_incons_collapsed = False
                    st.session_state.nma_incons_requested = True
                    st.session_state.panel_scroll_target = "nma_incons_panel_anchor"
        with dash_col:
            if st.button(_t("dashboard"), icon=":material/dashboard:", key="nma_open_dashboard", width="stretch"):
                st.session_state.nma_details_open = not st.session_state.nma_details_open
                if st.session_state.nma_details_open:
                    st.session_state.nma_details_collapsed = False
                    st.session_state.panel_scroll_target = "nma_details_panel_anchor"
                st.rerun()
        with more_col:
            if st.button(_t("more"), icon=":material/insights:", help=_t("more_help"),
                         key="nma_open_more", width="stretch"):
                st.session_state.nma_more_open = not st.session_state.nma_more_open
                if st.session_state.nma_more_open:
                    st.session_state.nma_more_collapsed = False
                    st.session_state.panel_scroll_target = "nma_more_panel_anchor"
        with ee_col:
            if st.button(_t("send_to_ee"), icon=":material/send:", help=_t("send_to_ee_help"),
                         key="nma_send_to_ee", width="stretch"):
                st.session_state.nma_ee_open = not st.session_state.nma_ee_open
                if st.session_state.nma_ee_open:
                    st.session_state.nma_ee_collapsed = False
                    st.session_state.panel_scroll_target = "nma_ee_panel_anchor"

    # ---- Sidebar ---------------------------------------------------------
    languages = ctx["LANGUAGES"]
    with st.sidebar:
        language_name = st.selectbox(
            "Language", list(languages),
            index=list(languages.values()).index(st.session_state.language),
            key="nma_language_choice",
        )
        if languages[language_name] != st.session_state.language:
            st.session_state.language = languages[language_name]
            st.rerun()

        with _nma_setting_panel("nma", "NMA settings"):
            with st.expander(T["file"], expanded=True):
                uploaded = None if os.environ.get("DECIDE_DESKTOP") == "1" else st.file_uploader(_t("open_hint"), type=["decide", "json", "csv"], key="nma_upload")
                open_clicked = uploaded is not None and st.button(_t("open_file"), use_container_width=True, key="nma_open_file")
                if st.button(_t("load_example"), icon=":material/science:", use_container_width=True, key="nma_load_example"):
                    _load_file([sup.default_document()])
                    st.rerun()
                # Import study data straight from a systematic-review tool's
                # export (RevMan .rm5 / CSV, Covidence CSV) -- separate from
                # Open above, which loads a saved DECIDE model or a generic
                # arm/contrast CSV. Available in the desktop app too, since
                # this is study data, not a model file.
                import_upload = st.file_uploader(
                    _t("import_hint"), type=["rm5", "csv"], key="nma_import_upload",
                )
                import_clicked = import_upload is not None and st.button(
                    _t("import_button"), icon=":material/download:", use_container_width=True, key="nma_import_button",
                )
                save_slot = st.container()
                report_slot = st.container()

            _render_outcomes_sidebar(document, version)

            with st.expander(_t("model_settings"), expanded=True):
                outcome_labels = {key: _t(f"outcome_{key}") for key in OUTCOMES}
                current_outcome = spec.get("outcome", "binary")
                spec["outcome"] = st.selectbox(
                    _t("outcome"), OUTCOMES, index=OUTCOMES.index(current_outcome) if current_outcome in OUTCOMES else 0,
                    format_func=lambda key: outcome_labels[key], key=f"nma_outcome_{version}")
                if spec["outcome"] in ARM_LEVEL_MEASURES:
                    allowed = list(ARM_LEVEL_MEASURES[spec["outcome"]])
                    current_measure = spec.get("measure")
                    spec["measure"] = st.selectbox(
                        _t("measure"), allowed,
                        index=allowed.index(current_measure) if current_measure in allowed else 0,
                        format_func=lambda key: MEASURE_LABELS.get(key, key),
                        disabled=len(allowed) == 1, key=f"nma_measure_arm_{version}_{spec['outcome']}")
                else:
                    current_measure = spec.get("measure", "HR")
                    spec["measure"] = st.selectbox(
                        _t("measure"), CONTRAST_MEASURES,
                        index=CONTRAST_MEASURES.index(current_measure) if current_measure in CONTRAST_MEASURES else 0,
                        key=f"nma_measure_{version}")
                engine_labels = {"frequentist": _t("frequentist"), "bayesian": _t("bayesian")}
                pending_engine = st.session_state.pop("nma_engine_pending", None)
                if pending_engine:  # switched from the MCMC panel's toggle
                    spec["engine"] = pending_engine
                    st.session_state.pop(f"nma_engine_{version}", None)
                spec["engine"] = st.segmented_control(
                    _t("engine"), ["frequentist", "bayesian"], default=spec.get("engine", "frequentist"),
                    format_func=lambda key: engine_labels[key], key=f"nma_engine_{version}") or "frequentist"
                effects_labels = {"fixed": _t("fixed"), "random": _t("random")}
                spec["effects"] = st.segmented_control(
                    _t("effects_model"), ["fixed", "random"], default=spec.get("effects", "random"),
                    format_func=lambda key: effects_labels[key], key=f"nma_effects_{version}") or "random"
                treatment_names = _current_treatment_names(version)
                if treatment_names:
                    current_reference = spec.get("reference")
                    spec["reference"] = st.selectbox(
                        _t("reference"), treatment_names,
                        index=treatment_names.index(current_reference) if current_reference in treatment_names else 0,
                        key=f"nma_reference_{version}")
                spec["small_values"] = "good" if st.toggle(
                    _t("lower_better"), value=spec.get("small_values", "good") == "good",
                    help=_t("lower_better_help"), key=f"nma_small_values_{version}") else "bad"
                settings_left, settings_right = st.columns(2)
                with settings_left:
                    spec["zero_correction"] = st.number_input(
                        _t("zero_correction"), min_value=0.0, max_value=1.0, step=0.1,
                        value=float(spec.get("zero_correction", 0.5)), key=f"nma_zero_{version}")
                with settings_right:
                    levels = [0.90, 0.95, 0.99]
                    current_level = float(spec.get("ci_level", 0.95))
                    spec["ci_level"] = st.selectbox(
                        _t("ci_level"), levels, index=levels.index(current_level) if current_level in levels else 1,
                        format_func=lambda value: f"{value:.0%}", key=f"nma_ci_{version}")

            bayes_on = spec.get("engine") == "bayesian"
            with st.expander(_t("mcmc"), expanded=bayes_on):
                bayes = spec.setdefault("bayes", {})
                use_bayes = st.toggle(_t("mcmc_use"), value=bayes_on, key=f"nma_mcmc_use_{version}_{bayes_on}")
                if use_bayes != bayes_on:
                    st.session_state.nma_engine_pending = "bayesian" if use_bayes else "frequentist"
                    st.rerun()
                exact_available = (spec.get("outcome"), spec.get("measure")) in EXACT_LIKELIHOOD
                likelihood_labels = {"exact": _t("lik_exact"), "normal": _t("lik_normal")}
                options = ["exact", "normal"] if exact_available else ["normal"]
                current_likelihood = bayes.get("likelihood", "exact")
                bayes["likelihood"] = st.selectbox(
                    _t("likelihood"), options,
                    index=options.index(current_likelihood) if current_likelihood in options else 0,
                    format_func=lambda key: likelihood_labels[key], disabled=not bayes_on,
                    help=_t("likelihood_help"), key=f"nma_mcmc_likelihood_{version}_{exact_available}")
                exact_on = bayes_on and bayes["likelihood"] == "exact"
                if bayes_on:
                    st.caption(_t("mcmc_note_exact") if exact_on else _t("mcmc_note"))
                else:
                    st.caption(_t("mcmc_soon"))
                mcmc_left, mcmc_right = st.columns(2)
                with mcmc_left:
                    bayes["chains"] = int(st.number_input(_t("chains"), min_value=1, max_value=8, step=1,
                                                          value=int(bayes.get("chains", 4)), disabled=not bayes_on,
                                                          key=f"nma_mcmc_chains_{version}"))
                    bayes["seed"] = int(st.number_input(_t("seed"), min_value=0, step=1, value=int(bayes.get("seed", 2026)),
                                                        disabled=not bayes_on, key=f"nma_mcmc_seed_{version}"))
                    bayes["warmup"] = int(st.number_input(_t("warmup"), min_value=100, max_value=20000, step=500,
                                                          value=int(bayes.get("warmup", 1000)), disabled=not exact_on,
                                                          key=f"nma_mcmc_warmup_{version}"))
                with mcmc_right:
                    bayes["draws"] = int(st.number_input(_t("draws"), min_value=100, max_value=20000, step=500,
                                                         value=int(bayes.get("draws", 1000)), disabled=not bayes_on,
                                                         key=f"nma_mcmc_draws_{version}"))
                    bayes["prior_sd"] = float(st.number_input(_t("prior_sd"), min_value=1.0, step=10.0,
                                                              value=float(bayes.get("prior_sd", 100.0)),
                                                              disabled=not bayes_on, key=f"nma_mcmc_prior_sd_{version}"))
                prior_labels = {"uniform": _t("tau_uniform"), "half-normal": _t("tau_half_normal"),
                                "lognormal": _t("tau_informative")}
                prior_scale, prior_note = nma_priors.scale_for(spec.get("outcome", "binary"), spec.get("measure", "OR"))
                prior_options = ["uniform", "half-normal"] + (["lognormal"] if prior_scale else [])
                current_prior = bayes.get("tau_prior", "uniform")
                if current_prior not in prior_options:
                    current_prior = "uniform"
                prior_col, value_col = st.columns([1.4, 1])
                with prior_col:
                    bayes["tau_prior"] = st.selectbox(
                        _t("tau_prior"), prior_options, index=prior_options.index(current_prior),
                        format_func=lambda key: prior_labels[key], disabled=not bayes_on,
                        help=_t("tau_prior_help"), key=f"nma_mcmc_tau_prior_{version}_{prior_scale}")
                with value_col:
                    if bayes["tau_prior"] == "lognormal":
                        st.caption(nma_priors.SOURCES[prior_scale])
                    elif bayes["tau_prior"] == "uniform":
                        bayes["tau_max"] = float(st.number_input(_t("tau_upper"), min_value=0.1, step=0.5,
                                                                 value=float(bayes.get("tau_max", 5.0)),
                                                                 disabled=not bayes_on, key=f"nma_mcmc_tau_max_{version}"))
                    else:
                        bayes["tau_scale"] = float(st.number_input(_t("tau_scale"), min_value=0.01, step=0.1,
                                                                   value=float(bayes.get("tau_scale", 0.5)),
                                                                   disabled=not bayes_on, key=f"nma_mcmc_tau_scale_{version}"))
                if bayes["tau_prior"] == "lognormal":
                    outcomes = [nma_priors.GENERAL] + (nma_priors.BINARY_OUTCOMES if prior_scale == "binary"
                                                       else nma_priors.CONTINUOUS_OUTCOMES)
                    comparisons = [nma_priors.GENERAL] + (nma_priors.COMPARISONS if prior_scale == "binary"
                                                          else nma_priors.CONTINUOUS_COMPARISONS)
                    bayes["tau_outcome"] = st.selectbox(
                        _t("tau_outcome_type"), outcomes,
                        index=outcomes.index(bayes.get("tau_outcome")) if bayes.get("tau_outcome") in outcomes else 0,
                        disabled=not bayes_on, key=f"nma_mcmc_tau_outcome_{version}_{prior_scale}")
                    bayes["tau_comparison"] = st.selectbox(
                        _t("tau_comparison_type"), comparisons,
                        index=comparisons.index(bayes.get("tau_comparison")) if bayes.get("tau_comparison") in comparisons else 0,
                        disabled=not bayes_on, key=f"nma_mcmc_tau_comparison_{version}_{prior_scale}")
                    chosen = _informative_tau(spec)
                    if chosen is not None:
                        bayes.update(tau_mu=chosen["mu"], tau_sigma=chosen["sigma"], tau_df=chosen["df"],
                                     tau_prior_label=chosen["label"])
                        st.caption(chosen["rescaled"].describe())
                        if not chosen["entry"].specific and (bayes.get("tau_outcome", nma_priors.GENERAL) != nma_priors.GENERAL
                                                            or bayes.get("tau_comparison", nma_priors.GENERAL) != nma_priors.GENERAL):
                            st.caption("ℹ️ " + _t("tau_cell_missing"))
                        if chosen["note"]:
                            st.caption("⚠️ " + chosen["note"])
                else:
                    for key in ("tau_mu", "tau_sigma", "tau_df", "tau_prior_label"):
                        bayes.pop(key, None)

                class_counts: dict[str, int] = {}
                for row in document.get("treatments", []):
                    label = str(row.get("class") or "").strip()
                    if label:
                        class_counts[label] = class_counts.get(label, 0) + 1
                class_active = exact_on and any(n >= 2 for n in class_counts.values())
                class_options = ["none", "common", "exchangeable"]
                class_labels = {"none": _t("class_none"), "common": _t("class_common"), "exchangeable": _t("class_exchangeable")}
                current_class = bayes.get("class_model", "none") if class_active else "none"
                bayes["class_model"] = st.selectbox(
                    _t("class_model"), class_options,
                    index=class_options.index(current_class if current_class in class_options else "none"),
                    format_func=lambda key: class_labels[key], disabled=not class_active, help=_t("class_model_help"),
                    key=f"nma_mcmc_class_model_{version}")
                if bayes["class_model"] == "exchangeable":
                    bayes["class_sd_scale"] = float(st.number_input(
                        _t("class_sd_scale"), min_value=0.01, step=0.25, value=float(bayes.get("class_sd_scale", 1.0)),
                        disabled=not bayes_on, key=f"nma_mcmc_class_sd_{version}"))
                if bayes_on and not class_active:
                    st.caption(_t("class_unavailable"))

            with st.expander(_t("cmp_tab"), expanded=False):
                comp_kind = st.radio(
                    _t("cmp_model"), ["freq", "bayes"], horizontal=True,
                    index=1 if spec.get("component_model") == "bayes" else 0, key=f"nma_cmp_model_{version}",
                    format_func=lambda k: _t("cmp_model_freq") if k == "freq" else _t("cmp_model_bayes"))
                spec["component_model"] = comp_kind
                comp_parts: list[str] = []
                for name, text in (document.get("components") or {}).items():
                    for part in nma_components.parse_components(text):
                        if part not in comp_parts:
                            comp_parts.append(part)
                comp_pairs = [f"{a}:{b}" for i, a in enumerate(comp_parts) for b in comp_parts[i + 1:]]
                saved_pairs = [item for item in spec.get("component_interactions", []) if item in comp_pairs]
                if comp_kind == "bayes":
                    spec["component_interactions"] = st.multiselect(
                        _t("cmp_interactions"), comp_pairs, default=saved_pairs, key=f"nma_cmp_interactions_{version}")
                elif not comp_parts:
                    st.caption(_t("cmp_none"))

            with st.expander(f"{_t('treatments')} ({len(treatment_names)})", expanded=False):
                st.caption(_t("treatments_help"))
                treatments_df = _table_editor(
                    "nma_treatments_table", st.session_state.nma_base_treatments, sup.TREATMENT_COLUMN_SPEC,
                    sup.TREATMENT_COLUMNS, [], version, max_height=320)

    # ---- Main area -------------------------------------------------------
    _render_autosave_banner()
    banner_slot = st.container()
    check_slot = st.container()
    data_col, network_col = st.columns([1, 1], gap="medium")

    with data_col:
        with st.container(border=True, key="nma_data_panel"):
            if len(st.session_state.nma_analyses) > 1:
                title = sup.analysis_title(document, st.session_state.nma_active)
                st.markdown(f"#### {_t('study_data')} · {html_lib.escape(title)}")
            else:
                st.markdown(f"#### {_t('study_data')}")
            arm_tab, contrast_tab, chars_tab = st.tabs([_t("arm_level"), _t("contrast_level"), _t("characteristics")])
            outcome = spec.get("outcome", "binary")
            arm_fields = sup.ARM_FIELDS_BY_OUTCOME.get(outcome, ["events", "n"])
            arm_spec = [column for column in sup.ARM_COLUMN_SPEC
                        if column["key"] in ("study", "treatment", *arm_fields, "source")]
            with arm_tab:
                arms_df = _table_editor("nma_arms_table", st.session_state.nma_base_arms, arm_spec,
                                        sup.ARM_COLUMNS, sup.NUMERIC_ARM_COLUMNS, version)
            with contrast_tab:
                st.caption(_t("contrast_help"))
                contrasts_df = _table_editor("nma_contrasts_table", st.session_state.nma_base_contrasts,
                                             sup.CONTRAST_COLUMN_SPEC, sup.CONTRAST_COLUMNS,
                                             sup.NUMERIC_CONTRAST_COLUMNS, version)
            with chars_tab:
                st.caption(_t("characteristics_help"))
                chars_df = _table_editor("nma_chars_table", st.session_state.nma_base_chars,
                                         sup.characteristic_column_spec(document), sup.characteristic_columns(document),
                                         sup.numeric_characteristic_columns(document), version)
                _render_characteristic_variables(document, treatments_df, arms_df, contrasts_df, chars_df)
            with st.expander(_t("conv_title"), icon=":material/calculate:", expanded=False):
                if st.session_state.get("nma_conv_fill_message"):
                    st.success(st.session_state.pop("nma_conv_fill_message"))
                _render_conversions(spec, treatments_df, arms_df, contrasts_df, chars_df)

    # Assemble the document from the current tables.
    document["treatments"] = sup.treatments_from_frame(treatments_df)
    document["studies"] = sup.studies_from_frames(arms_df, contrasts_df)
    document["study_characteristics"] = sup.characteristics_from_frame(chars_df)
    st.session_state.nma_document = document

    parse_error: str | None = None
    model = summary = None
    errors: list = []
    try:
        model = resolve_nma(document)
        errors = validate_nma(model)
        summary = summarize_network(model)
    except ValueError as exc:
        parse_error = str(exc)

    # ---- Network panel ---------------------------------------------------
    with network_col:
        with st.container(border=True, key="nma_network_panel"):
            head_col, view_col = st.columns([1, 1], vertical_alignment="center")
            with head_col:
                st.markdown(f"#### {_t('network')}")
            with view_col:
                st.session_state.nma_view = st.segmented_control(
                    _t("view"), ["2D", "3D"], default=st.session_state.nma_view,
                    key="nma_view_control", label_visibility="collapsed") or "2D"
            colour_col, size_col = st.columns([1.4, 1], vertical_alignment="bottom")
            colour_labels = {"class": _t("by_class"), "single": _t("single"), "custom": _t("custom")}
            with colour_col:
                st.session_state.nma_color_mode = st.segmented_control(
                    _t("node_colour"), ["class", "single", "custom"], default=st.session_state.nma_color_mode,
                    format_func=lambda key: colour_labels[key], key="nma_color_control") or "class"
            size_labels = {"patients": _t("patients"), "studies": _t("studies"), "equal": _t("equal")}
            with size_col:
                st.session_state.nma_size_by = st.selectbox(
                    _t("size_by"), ["patients", "studies", "equal"],
                    index=["patients", "studies", "equal"].index(st.session_state.nma_size_by),
                    format_func=lambda key: size_labels[key], key="nma_size_control")

            selected_edge = None
            if model is not None and summary is not None and summary.treatments:
                if st.session_state.nma_color_mode == "class":
                    legend = sup.class_legend(model)
                    st.markdown(" &nbsp; ".join(
                        f"<span style='display:inline-block;width:11px;height:11px;border-radius:6px;background:{color};"
                        f"margin-right:4px'></span><b>{html_lib.escape(label)}</b> <span style='color:#475569'>{html_lib.escape(', '.join(members))}</span>"
                        for label, color, members in legend), unsafe_allow_html=True)
                edge_keys = list(summary.edges)
                edge_choice = st.selectbox(
                    _t("selected_edge"), [None, *edge_keys],
                    format_func=lambda key: _t("no_edge") if key is None else
                    f"{key[0]} vs {key[1]} ({summary.edges[key].n_studies})",
                    key="nma_selected_edge")
                selected_edge = edge_choice
                payload = sup.network_payload(model, summary, st.session_state.nma_color_mode,
                                              st.session_state.nma_size_by, selected_edge)
                payload["legend"] = sup.figure_legend(model, st.session_state.nma_color_mode)
                payload["title"] = st.session_state.nma_document.get("name", "") or ""
                payload["filename"] = sup.figure_filename(payload["title"], st.session_state.nma_view)
                _render_network(payload, st.session_state.nma_view, 400, st.session_state.nma_layout_nonce)
                if selected_edge is not None:
                    edge = summary.edges[selected_edge]
                    via = sup.indirect_paths(summary, *selected_edge)
                    with st.container(key="nma_edge_info"):
                        st.markdown(f"**{selected_edge[0]} vs {selected_edge[1]}**")
                        st.markdown(
                            f"{_t('direct')}: {', '.join(edge.study_ids)} "
                            f"({edge.n_studies} · {int(edge.patients)} {_t('patients').lower()})")
                        st.markdown(f"{_t('indirect_via')}: {', '.join(via) if via else _t('none')}")

    # ---- Banner, check, file actions (need the assembled document) -------
    unknown = sorted({error.message.split("'")[1] for error in errors
                      if error.code == "unknown_treatment" and "'" in error.message})
    with banner_slot:
        if parse_error:
            st.error(f"{_t('parse_error')}: {parse_error}")
        elif errors:
            lines = "\n".join(f"- {error.message}" for error in errors[:10])
            more = f"\n- … +{len(errors) - 10}" if len(errors) > 10 else ""
            st.error(f"**{_t('fix_errors')}**\n{lines}{more}")
            if unknown and st.button(_t("add_missing"), icon=":material/playlist_add:", key="nma_add_missing"):
                extra = pd.DataFrame([{"name": name, "aliases": "", "class": "", "color": ""} for name in unknown])
                _commit_tables(pd.concat([treatments_df, extra], ignore_index=True).astype(object), arms_df, contrasts_df,
                               chars_df)
                st.rerun()
        elif summary is not None:
            loops = [loop for loop in summary.loops if not loop.single_study]
            loop_text = f"{len(loops)} closed loop{'s' if len(loops) != 1 else ''}"
            if loops:
                loop_text += " (" + "; ".join("–".join(loop.treatments) for loop in loops[:3]) + ")"
            st.success(
                f"**{_t('network_ok')}** · {len(summary.treatments)} treatments · {len(model.studies)} studies · "
                f"{summary.n_arms} arms · {loop_text} · {len(summary.multi_arm_studies)} multi-arm"
            )
    if st.session_state.nma_check_open and summary is not None and model is not None:
        with check_slot:
            if os.environ.get("DECIDE_DESKTOP") == "1":  # desktop: centred window like the other result panels
                st.markdown(
                    """<style>
                    div.st-key-nma_check_modal { position: fixed !important; top: 50% !important; left: 50% !important;
                        transform: translate(-50%, -50%) !important; width: min(900px, 88vw) !important; box-sizing: border-box !important;
                        padding: 20px 30px 26px 30px !important; max-height: 86vh !important; overflow-y: auto !important;
                        z-index: 1000002 !important; background: #ffffff !important; border: 2px solid #1d4f9b; border-radius: 12px;
                        box-shadow: 0 0 0 100vmax rgba(15, 30, 60, 0.45), 0 12px 40px rgba(0,0,0,0.35); }
                    </style>""",
                    unsafe_allow_html=True,
                )
                with st.container(key="nma_check_modal"):
                    head, close = st.columns([14, 1], vertical_alignment="center")
                    head.markdown(f"### {_t('check_network')}")
                    if close.button("✕", key="nma_check_close_btn", help="Close"):
                        st.session_state.nma_check_open = False
                        st.rerun()
                    _render_check_body(summary, model)
            else:
                with st.container(border=True):
                    _render_check_body(summary, model)

    if open_clicked:
        try:
            if uploaded.name.lower().endswith((".decide", ".json")):
                _load_file(*sup.documents_from_file(json.load(uploaded)))
            else:
                kind, frame = sup.frames_from_csv(pd.read_csv(uploaded))
                new_arms = frame if kind == "arms" else arms_df
                new_contrasts = frame if kind == "contrasts" else contrasts_df
                refreshed_chars = sup.characteristics_frame({
                    "studies": sup.studies_from_frames(new_arms, new_contrasts),
                    "study_characteristics": sup.characteristics_from_frame(chars_df),
                })
                _commit_tables(treatments_df, new_arms, new_contrasts, refreshed_chars)
                if kind == "contrasts" and spec.get("outcome") != "contrast":
                    spec["outcome"] = "contrast"
                    spec["measure"] = spec.get("measure") if spec.get("measure") in CONTRAST_MEASURES else "HR"
            st.rerun()
        except (ValueError, json.JSONDecodeError) as exc:
            st.sidebar.error(str(exc))

    if import_clicked:
        try:
            from heor_tree_engine.nma_import import (
                NMAImportError,
                detect_import_format,
                parse_covidence_csv,
                parse_revman_csv,
                parse_revman_rm5,
                studies_to_document_rows,
            )

            raw = import_upload.getvalue()
            source = detect_import_format(import_upload.name, raw)
            if source == "rm5":
                imported = parse_revman_rm5(raw)
            elif source == "revman_csv":
                imported = parse_revman_csv(raw)
            else:
                imported = parse_covidence_csv(raw)
            rows = [
                {"study": study["id"], "treatment": arm.get("treatment", ""),
                 "events": arm.get("events"), "n": arm.get("n"),
                 "mean": arm.get("mean"), "sd": arm.get("sd"),
                 "exposure": arm.get("exposure"), "source": study.get("source", "")}
                for study in imported for arm in study.get("arms", [])
            ]
            new_arms = pd.DataFrame(rows, columns=sup.ARM_COLUMNS)
            for column in sup.NUMERIC_ARM_COLUMNS:
                new_arms[column] = pd.to_numeric(new_arms[column], errors="coerce").astype("float64")
            refreshed_chars = sup.characteristics_frame({
                "studies": sup.studies_from_frames(new_arms, contrasts_df),
                "study_characteristics": sup.characteristics_from_frame(chars_df),
            })
            _commit_tables(treatments_df, new_arms, contrasts_df, refreshed_chars)
            # Pre-set the outcome to match what was imported (binary vs
            # continuous) so the measure selector starts on a legal value.
            outcome = studies_to_document_rows(imported)["outcome"]
            if outcome in ("binary", "continuous") and spec.get("outcome") != outcome:
                spec["outcome"] = outcome
                allowed = list(ARM_LEVEL_MEASURES.get(outcome, []))
                if allowed and spec.get("measure") not in allowed:
                    spec["measure"] = allowed[0]
            st.sidebar.success(_t("import_done").format(n=len(imported)))
            st.rerun()
        except (NMAImportError, ValueError) as exc:
            st.sidebar.error(str(exc))

    # ---- Run -------------------------------------------------------------
    if st.session_state.nma_run_requested:
        st.session_state.nma_run_requested = False
        if model is not None and not errors:
            st.session_state.nma_result = run_nma_frequentist(model)
            st.session_state.nma_result_hash = sup.document_hash(document)
            if st.session_state.nma_result.errors:
                st.error("\n".join(error.message for error in st.session_state.nma_result.errors))
                st.session_state.nma_result = None
            if spec.get("engine") == "bayesian" and st.session_state.nma_result is not None:
                st.session_state.nma_bayes = _run_bayes(model, spec)
                st.session_state.nma_bayes_hash = sup.bayes_hash(document)
            if st.session_state.nma_incons_open:
                st.session_state.nma_incons_requested = True
    if st.session_state.nma_incons_requested:
        st.session_state.nma_incons_requested = False
        if model is not None and not errors:
            st.session_state.nma_incons_result = run_inconsistency(model, method=_split_method())
            st.session_state.nma_incons_hash = _incons_hash()

    render_panel = ctx["render_collapsible_panel"]
    if st.session_state.nma_results_open:
        render_panel(_t("results_title"), "nma_results_open", "nma_results_collapsed",
                     "nma_results_panel_anchor", _render_results_body)
    if st.session_state.nma_incons_open:
        render_panel(_t("incons_title"), "nma_incons_open", "nma_incons_collapsed",
                     "nma_incons_panel_anchor", _render_inconsistency_body)
    if st.session_state.nma_more_open:
        render_panel(_t("more_title"), "nma_more_open", "nma_more_collapsed", "nma_more_panel_anchor",
                     _render_more_body)
    if st.session_state.nma_ee_open:
        render_panel(_t("ee_title"), "nma_ee_open", "nma_ee_collapsed", "nma_ee_panel_anchor", _render_ee_body)
    if st.session_state.nma_details_open:
        render_panel(_t("details_title"), "nma_details_open", "nma_details_collapsed",
                     "nma_details_panel_anchor", _render_details_body)

    # Filled last so the file includes edits made in the panels on this run
    # (e.g. CINeMA judgements).
    with save_slot:
        data = json.dumps(sup.bundle_documents(_all_documents(), st.session_state.nma_active),
                          indent=2, ensure_ascii=False)
        _nma_model_files().render_panel()
        _autosave(data)
    with report_slot:
        _render_report_export(model, summary)
