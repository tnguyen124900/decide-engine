from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image
from reportlab.platypus import (
    BaseDocTemplate,
    HRFlowable,
    PageBreak,
    Paragraph,
    PageTemplate,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

OUTPUT = Path(__file__).parent.parent / "static" / "DECIDE Manual.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
BLUE = colors.HexColor("#1d4f9b")
LIGHT_BLUE = colors.HexColor("#eaf2ff")

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ManualTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=BLUE,
        spaceAfter=18,
    )
)
styles.add(ParagraphStyle(name="ManualHeading", parent=styles["Heading2"], textColor=BLUE))
styles.add(ParagraphStyle(name="ManualVersion", parent=styles["Heading2"], textColor=BLUE))
styles.add(ParagraphStyle(name="ManualPart", parent=styles["Heading1"], textColor=BLUE, spaceBefore=6, spaceAfter=10))
styles.add(ParagraphStyle(name="ManualBody", parent=styles["BodyText"], leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name="ManualCode", parent=styles["Code"], fontName="Courier", fontSize=8.5, leading=11))
styles.add(ParagraphStyle(name="ManualContentsTitle", parent=styles["Heading1"], textColor=BLUE, spaceAfter=14))


class ManualDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate with clickable bookmarks and a generated contents table."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph) or flowable.style.name not in ("ManualHeading", "ManualPart"):
            return
        title = flowable.getPlainText()
        key = f"manual-heading-{self.page}-{id(flowable)}"
        self.canv.bookmarkPage(key)
        level = 0 if flowable.style.name == "ManualPart" else 1
        self.canv.addOutlineEntry(title, key, level=level, closed=False)
        self.notify("TOCEntry", (level, title, self.page, key))


def p(text: str, style: str = "ManualBody") -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), styles[style])


def table(headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    data = [[p(f"<b>{value}</b>") for value in headers]]
    data.extend([[p(value) for value in row] for row in rows])
    result = Table(data, repeatRows=1, colWidths=col_widths or [1.65 * inch, 5.55 * inch])
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7d0f4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return result


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#57606a"))
    canvas.drawString(0.55 * inch, 0.45 * inch, "Copyright 2026 Trang Nguyen - DECIDE. All rights reserved.")
    canvas.drawRightString(7.5 * inch, 0.45 * inch, f"DECIDE Manual | {doc.page}")
    canvas.restoreState()


story = [
    Image(str(Path(__file__).parent.parent / "assets" / "decide-logo.png"), width=2.3 * inch, height=0.95 * inch),
    Spacer(1, 24),
    p("DECIDE Manual", "ManualTitle"),
    p("Version 1.2", "ManualVersion"),
    p("1 October 2026", "ManualBody"),
    Spacer(1, 18),
    HRFlowable(width="100%", thickness=1, color=BLUE),
    Spacer(1, 12),
    PageBreak(),
    p("Contents", "ManualContentsTitle"),
    TableOfContents(),
    PageBreak(),

    # =====================================================================
    # PART A -- The DECIDE Expression Language
    # =====================================================================
    p("Part A -- The DECIDE Expression Language", "ManualPart"),
    p("The knowledge in this part is shared by every model type in DECIDE (Decision Tree, Markov, and future model types): the same expression evaluator, the same functions, and the same Parameters/Variables/Distributions/Tables data model underlie every Cost, Effect, Probability, and definition field in the app."),
    Spacer(1, 8),

    p("Expression engine", "ManualHeading"),
    p("DECIDE expressions use Python-compatible semantics. The same expression rules are available in decision trees and Markov models."),
    p("Parameters are numeric inputs. Variables are named expressions that may reference parameters and other variables."),
    p("Example: <font name='Courier'>prob_AE = RateToProb(rate_AE_3yr, 1, 1/4)</font> with <font name='Courier'>rate_AE_3yr = 0.5</font> gives approximately <font name='Courier'>0.1175030974</font>.<br/><br/>"),

    p("Operators", "ManualHeading"),
    table(["Operator", "Meaning"], [
        ["+  -  *  /", "Addition, subtraction, multiplication, division"],
        ["**", "Exponentiation"],
        ["%  //", "Modulus and floor division"],
        ["==  !=  <  <=  >  >=", "Comparisons"],
        ["and  or  not", "Logical operators"],
        ["( ... )", "Grouping and precedence"],
    ]),
    Spacer(1, 14),
    p("Constants: <font name='Courier'>e</font> is Euler's number and <font name='Courier'>pi</font> is pi. Trigonometric functions use radians."),
    p("Scalar functions", "ManualHeading"),
    table(["Function", "Meaning"], [
        ["abs(x)", "Absolute value"],
        ["arccos(x), arcsin(x), arctan(x)", "Inverse trigonometric functions"],
        ["bound(x, (a,b)) or bound(x, [a,b])", "Clamps x into the closed interval [a, b]. The bracket style does not change the result; mixed forms such as (a,b] are not valid syntax."],
        ["cbrt(x)", "Cube root"],
        ["ceil(x), floor(x)", "Ceiling and floor"],
        ["comb(n, k)", "Binomial coefficient; n >= k >= 0"],
        ["cos(x), sin(x), tan(x)", "Trigonometric functions"],
        ["cosh(x), sinh(x), tanh(x)", "Hyperbolic functions"],
        ["erf(x), invErf(x)", "Error function and inverse error function"],
        ["exp(x)", "e**x"],
        ["factorial(n)", "Factorial for a non-negative integer"],
        ["gamma(x), lgamma(x)", "Gamma and log-Gamma functions"],
        ["hypot(x,y)", "sqrt(x**2 + y**2)"],
        ["if(condition, a, b)", "Return a if condition is true, otherwise b"],
        ["ln(x)", "Natural logarithm"],
        ["log(x,b), log10(x)", "Logarithm to base b and base 10"],
        ["logit(p), logis(x)", "Log odds and logistic function"],
        ["max(a,b), min(a,b)", "Maximum and minimum"],
        ["round(x), round(x,n)", "Python-compatible rounding"],
        ["sign(x)", "-1, 0, or 1 according to the sign of x"],
        ["sqrt(x)", "Square root"],
    ]),
    PageBreak(),
    p("Probability and summary functions", "ManualHeading"),
    table(["Function", "Meaning"], [
        ["ProbToProb(p,t1,t2)", "1 - (1-p)**(t2/t1)"],
        ["ProbToRate(p,t1,t2)", "-ln(1-p) * t2/t1"],
        ["RateToProb(r,t1,t2)", "1 - exp(-r * t2/t1)"],
        ["mean(x1,...,xn)", "Arithmetic mean"],
        ["prod(x1,...,xn)", "Product"],
        ["quantile({x1,...,xn},q)", "Linear-interpolated quantile, 0 <= q <= 1"],
        ["sd(x1,...,xn)", "Sample standard deviation, denominator n-1"],
        ["sum(x1,...,xn)", "Sum"],
        ["var(x1,...,xn)", "Sample variance, denominator n-1"],
    ]),
    Spacer(1, 14),
    p("Distribution functions", "ManualHeading"),
    p("The distribution argument is the name of a distribution configured in the Distribution section. Its type and parameters are not repeated in the function call."),
    table(["Function", "Meaning"], [
        ["DistMean(distribution)", "Mean of the configured distribution"],
        ["DistSample(distribution)", "Random sample from the configured distribution"],
        ["DistCDF(distribution,s)", "Cumulative probability P(X <= s)"],
        ["DistPPF(distribution,p)", "Value s whose cumulative probability is p"],
        ["DistSampleCond(distribution, interval)", "Conditional random sample in an interval such as [lower, upper] or (lower, upper)"],
        ["DistToProb(distribution,t1,t2)", "Conditional transition probability during [t1,t1+t2] given no event before t1"],
        ["DistToTime(distribution,t1)", "Random remaining time until the event for someone still event-free at t1 (t1 optional, default 0); measured from t1, like t2 in DistToProb"],
        ["DistTrim(distribution,[min,max])", "Random sample trimmed to min <= X <= max"],
        ["DistSurv(distribution,s)", "Survival probability, 1 - DistCDF(distribution,s)"],
        ["DistHazard(distribution,s)", "Hazard, PDF(s) / survival(s)"],
    ]),
    Spacer(1, 10),
    p("DistToProb is calculated from the conditional survival S(t1+t2)/S(t1), not a plain CDF or PDF. DistToTime solves the same formula for time instead of probability, so the two always agree: the share of DistToTime(dist, t1) draws that are <= t2 equals DistToProb(dist, t1, t2)."),
    Spacer(1, 10),
    p("Distribution types and parameters", "ManualHeading"),
    p("<b>DECIDE follows the parameterization of the R package flexsurv</b> (for most families also the one of R's own d/p/q/r functions, such as pweibull and qgamma). Values fitted with flexsurv, or reported in a paper that uses flexsurv, can be typed straight into the Distribution table. The Gamma distribution uses <b>shape and rate</b>, as flexsurv does (scale = 1/rate)."),
    table(["Distribution (parameters)", "Survival function S(t) and notes (t in the distribution's time unit)"], col_widths=[2.3 * inch, 4.9 * inch], rows=[
        ["Exponential (rate)", "S(t) = exp(-rate*t); mean 1/rate"],
        ["Weibull (shape, scale)", "S(t) = exp(-(t/scale)^shape); mean scale * Gamma(1 + 1/shape)"],
        ["Gamma (shape, rate)", "mean shape/rate, variance shape/rate^2; scale = 1/rate"],
        ["Log-normal (meanlog, sdlog)", "S(t) = 1 - Phi((ln t - meanlog)/sdlog)"],
        ["Log-logistic (shape, scale)", "S(t) = 1/(1 + (t/scale)^shape); median = scale"],
        ["Gompertz (shape, rate)", "hazard h(t) = rate*exp(shape*t); S(t) = exp(-(rate/shape)*(exp(shape*t) - 1))"],
        ["Generalized gamma (mu, sigma, Q)", "Prentice (flexsurv gengamma) parameterization; Q = 0 is the log-normal"],
    ]),
    Spacer(1, 10),
    p("<b>Time unit.</b> DECIDE never converts time units. The distribution's parameters must be in the same unit as the arguments of DistToProb, DistToTime, DistSurv and so on, and as the model cycle (for example, a Weibull fitted in years used with monthly cycles needs DistToProb(d, t1/12, 1/12))."),
    p("<b>Entering values reported in other conventions.</b> The same name can mean different parameters in different software, so convert before typing:"),
    table(["Source / reported as", "Enter in DECIDE"], col_widths=[3.5 * inch, 3.7 * inch], rows=[
        ["R pweibull, flexsurvreg(dist=\"weibull\"): shape, scale", "As reported"],
        ["flexsurvreg(dist=\"weibullPH\"), Stata streg weibull (proportional hazards): cumulative hazard lambda * t^gamma", "shape = gamma, scale = lambda^(-1/gamma)"],
        ["Paper or software with cumulative hazard (lambda * t)^gamma (for example the DARTH time-dependent Sick-Sicker tutorial)", "shape = gamma, scale = 1/lambda"],
        ["survival::survreg Weibull (accelerated failure time): Intercept, Log(scale)", "scale = exp(Intercept), shape = 1 / survreg scale"],
        ["survreg log-logistic: Intercept, scale", "scale = exp(Intercept), shape = 1 / survreg scale"],
        ["survreg log-normal: Intercept, scale", "meanlog = Intercept, sdlog = scale"],
        ["Exponential given as a mean time theta", "rate = 1/theta"],
        ["Gamma given as shape k and scale theta", "shape = k, rate = 1/theta"],
        ["Gamma given as mean m and standard deviation s", "shape = (m/s)^2, rate = m/s^2"],
        ["flexsurvreg(dist=\"gengamma.orig\"): shape b, scale a, k", "Q = 1/sqrt(k), sigma = 1/(b*sqrt(k)), mu = ln(a) + ln(k)/b"],
    ]),
    Spacer(1, 10),
    p("With flexsurv output, use the natural-scale estimates (est in res), not the log-scale ones (res.t), and with covariates compute the parameter at the covariate values of the group being modelled before typing it. The Weibull is the most common source of confusion: a shape of 1.1 with lambda = 0.08 gives scale = 12.5 when the cumulative hazard is (lambda*t)^gamma but scale = 9.94 when it is lambda*t^gamma, and the one-cycle event probability at time 0 is 0.060 versus 0.077."),
    p("<b>Gamma and older models.</b> Earlier versions of DECIDE stored the Gamma distribution as k (shape) and theta (scale). Such models are converted when the Distribution row is loaded: k becomes shape and theta becomes rate = 1/theta (an expression is wrapped as 1/(expression)), and the distribution, and every result that depends on it, is unchanged. A PSA row that targets a Gamma slot by its old name theta must be pointed at rate again."),
    p("<b>Other distributions</b> (parameters as shown in the Distribution table):"),
    table(["Distribution (parameters)", "Definition"], col_widths=[2.9 * inch, 4.3 * inch], rows=[
        ["Beta (a, b)", "R shape1, shape2; mean a/(a+b)"],
        ["Normal (mu, sigma)", "Standard deviation, not variance"],
        ["Truncated Normal (mu, sigma, a, b)", "mu and sigma describe the untruncated normal; a, b are the bounds"],
        ["Uniform (a, b)", "Minimum and maximum"],
        ["Triangular (a, b, c)", "Minimum, mode, maximum"],
        ["PERT (a, b, c)", "Minimum, most likely, maximum; Beta with weight 4"],
        ["Poisson (lambda)", "Mean count"],
        ["Binomial (n, p); Bernoulli (p)", "Number of trials and probability; probability of success"],
        ["Geometric (p)", "Number of trials until the first success (starts at 1)"],
        ["Negative Binomial (r, p)", "Number of failures before the r-th success"],
        ["Discrete Uniform (a, b)", "Integers from a to b, both included"],
        ["Hypergeometric (w, b, n)", "White, black, number drawn"],
        ["Categorical (p)", "List of probabilities that sums to 1"],
        ["Student's t (nu); Chi-Square (k)", "Standard t (location 0, scale 1); degrees of freedom"],
        ["Cauchy (x0, gamma); Half-Cauchy (gamma)", "Location and scale; scale"],
        ["Half-Normal (sigma); Laplace (mu, b); Logistic (mu, s)", "Scale; location and scale; location and scale"],
        ["Gumbel (mu, beta)", "Maximum-extreme (right-skewed) form"],
        ["Pareto (alpha, k)", "Shape, minimum (type I)"],
        ["Zipf (s, n)", "Exponent, number of elements"],
    ]),
    Spacer(1, 10),
    p("A life-table column is a probability or a rate depending on its label: qx is an annual probability and is used directly, mx is a rate and goes through RateToProb."),
    Spacer(1, 10),
    p("Tables and Lookup", "ManualHeading"),
    p("A Table is a named grid of reference data, entered directly or uploaded from a CSV or Excel file. Lookup syntax:"),
    table(["Form", "Reads"], [
        ["table_name[lookup_value]", "Default result column"],
        ["table_name[lookup_value, result_column]", "Named/indexed result column"],
        ["table_name[lookup_value, lookup_column, result_column]", "Named/indexed lookup and result columns"],
        ["table_name[..., method]", 'Optional trailing method: "exact" | "truncate" | "linear" (default) | "spline"'],
    ]),
    Spacer(1, 10),
    p("Table and Lookup examples", "ManualHeading"),
    table(["Example", "Meaning"], [
        ['t_mortality[Current_age, "Age", "Mortality"]', "Age-banded mortality, linear interpolation"],
        ['t_mortality[_cycle, "Cycle", "Prob"]', "A Markov transition probability that changes every cycle"],
        ['t_cost[Current_age, 0, 1, "exact"]', "Exact age-band cost lookup by column index"],
        ['RateToProb(t_hazard[_cycle, "Cycle", "Rate"], 1, 1)', "Lookup combined with a rate/probability conversion"],
    ]),
    Spacer(1, 10),
    p("TableToProb and TableToTime", "ManualHeading"),
    p("The table counterparts of DistToProb/DistToTime, for survival or mortality tables (e.g. a life table) instead of parametric distributions. TableToProb(table, t1, t2[, value_column[, type]]) returns the probability of the event in (t1, t1+t2] for someone event-free at t1; TableToTime(table, t1[, value_column[, type]]) returns the random remaining time from t1. The type argument accepts \"conditional\" (default -- q(x)), \"probability\", or \"cumulative\", depending on the input table's form."),
    Spacer(1, 10),
    p("Parameters vs. Variables", "ManualHeading"),
    p("<b>Parameter</b>: a root input, with name / base / low / high / source. Base can be a number or a formula referencing other Parameters. Low/High feed Sensitivity Analysis and Model Calibration bounds. A Parameter can have a PSA distribution attached."),
    p("<b>Variable</b>: a quantity computed from other Parameters/Variables via a definition (formula). In a Markov model, a Variable's definition can depend on <font name='Courier'>_cycle</font>/<font name='Courier'>_tenure</font> (see Part C), so its value can change by cycle; a Decision Tree Variable always evaluates with <font name='Courier'>_cycle=0</font>, <font name='Courier'>_tenure=1</font>, so its value is fixed."),
    p("Parameters and Variables share one namespace -- a Variable's name cannot duplicate a Parameter's name. Use <font name='Courier'>prob_AE</font> in a decision-tree branch probability, terminal cost/effect, Markov initial probability, state cost/effect, or transition probability."),
    p("<b>Renaming is automatic.</b> Rename a Parameter, Variable, Distribution, Table or Tracker in its input table and every place that uses the old name is rewritten to the new one -- canvas formulas (branches, states, transitions), other Variables' definitions, Distribution parameters, trackers and rates, PSA setup and Sensitivity/Calibration selections -- in Decision Tree, Markov, PSM, DES, Microsimulation and ABM alike. Text inside quotes (a table column such as <font name='Courier'>tbl[\"age\"]</font>) and longer names that merely contain the old one are never touched. If the new name is already taken, nothing is rewritten and the duplicate-name warning shows."),
    PageBreak(),

    # =====================================================================
    # PART B -- Decision Tree
    # =====================================================================
    p("Part B -- Decision Tree", "ManualPart"),

    p("File panel", "ManualHeading"),
    p("Decision Tree, Markov, PSM, ABM and DynaTrans share the same File panel; NMA has Save / Save as / Open with its own autosave."),
    table(["Control", "What it does"], [
        ["Save", "Writes the model straight to Folder/Filename and overwrites an existing file without asking. Once a file is saved or opened, Save keeps updating that file."],
        ["Save as", "Downloads a copy through the browser, where you choose the name and folder."],
        ["Folder / Saved models / Open saved model", "Choose the folder Save and Open use (remembered for next time; each model type defaults to its own subfolder of models/) and open any model saved there."],
        ["Autosave and Restore", "Every change is autosaved in the background (five rotating backups in the autosave folder). If the app is reopened with unsaved work from an earlier session, a Restore banner offers to bring it back. There is no unsaved-changes warning."],
    ]),
    Spacer(1, 10),

    p("Decision Tree Editor", "ManualHeading"),
    p("The Tree Editor draws the tree on a canvas. Branch labels, probabilities, costs and effects are edited on the canvas itself; node names are not shown. Strategies are the branches leaving the Decision node."),
    table(["Feature", "How it works"], [
        ["Strategy branches", "A Decision-node branch only carries a label: it takes no probability, no $ and no E input. Cost and effect belong to the chance branches and terminals below it."],
        ["Path results on the canvas", "Beside each terminal the canvas shows that path's probability p, total cost and total effect (branch increments plus the terminal value, with the terminal discounted by its year). Turn it on or off with 'Show path results'."],
        ["Strategy totals", "Under each strategy node the canvas shows the base-case expected totals, labelled total_ plus the outcome symbol (for example total_$ and total_E). They equal the sum of p x value over that strategy's paths."],
        ["Base case results", "The Paths table lists every path by branch labels (Strategy → label → label), with undiscounted and discounted cost/effect. Paths CSV and Trace CSV downloads give the step-by-step cumulative trace; a warning appears if the paths do not reconcile with the engine totals."],
        ["Cost-only models", "Delete the Effect row in Outcomes (Model properties) to run a cost-only model: effect inputs are hidden and zeroed, and restored if an Effect row is added again. Analysis then skips the effect picker."],
        ["Text style", "The toolbar above the canvas sets font, size and Bold / Italic / Underline separately for each text type (branch label, probability, cost, effect, path results, strategy totals). Settings are per text type, not synced."],
        ["Save, Save as, Open", "Save writes the current model straight to Folder/Filename, overwriting an existing file without asking. Save as downloads a copy through the browser. Open loads a model from the Saved models list."],
        ["Autosave and Restore", "The model is autosaved in the background. If the app is reopened with an unsaved earlier session, a Restore banner offers to bring that work back."],
    ]),
    Spacer(1, 10),

    PageBreak(),
    p("Model Calibration", "ManualHeading"),
    p("Model Calibration adjusts a model's own Parameters and Variables so its output matches real-world data -- a known outcome rate, or, for Markov models, a Kaplan-Meier-style curve of state occupancy over time. It works the same way across model types: a sidebar button opens a panel below the model's Parameters/Variables/Distributions/Tables cards -- pick which inputs may move, enter the data to match, choose an algorithm, run it, then apply or discard. The Markov-specific target-row shape is covered in Part C; everything else on this page applies to both model types identically."),
    table(["Model", "Target row columns", "Notes"], [
        ["Markov", "Cycle, State, Value, Weight", "State is picked from the states of the strategy being calibrated. Value is the observed occupancy fraction for that state at that cycle."],
        ["Decision tree", "Metric, Target ID, Value, Weight", "Metric is total_cost, total_effect, or terminal_probability. Target ID names a terminal node and only applies to terminal_probability."],
    ]),
    Spacer(1, 10),
    p("Algorithms", "ManualHeading"),
    table(["Algorithm", "How it works / trade-off"], [
        ["Local Optimization", "Steps from each input's base value toward the smallest mismatch until it converges. Fastest, but can settle in a local best fit."],
        ["Random Search", "Draws random candidate combinations within Low/High bounds, keeps the best, optionally refines with Local Optimization."],
        ["Grid Search", "Same idea as Random Search but over an evenly spaced grid. Cost grows fast with the number of calibrated inputs."],
        ["Bayesian (SIR)", "Sampling-Importance-Resampling: draws from each input's prior, scores by fit, resamples into a posterior."],
        ["Bayesian (MCMC)", "Metropolis-Hastings chain of proposed candidates, accepted or rejected by fit, approximating the posterior."],
    ]),
    Spacer(1, 10),
    p("Bayesian Calibration", "ManualHeading"),
    p("SIR and MCMC report a posterior mean, standard deviation, and 95% credible interval for every calibrated input, instead of one best-fit point. The result panel also reports a diagnostic -- effective sample size for SIR, acceptance rate (aim for roughly 15–50%) for MCMC -- alongside a converged/needs-review flag."),
    p("Applying a result", "ManualHeading"),
    p("'Apply calibrated values' (Local/Random/Grid) or 'Apply posterior mean as base case' (Bayesian) writes the result into the model's real Parameters/Variables; 'Discard result' clears the run instead. A Bayesian result adds 'Apply posterior to PSA (recommended)', which fits a Beta distribution (when the posterior mean is strictly between 0 and 1) or a Normal distribution otherwise, and upserts it into that input's PSA Parameters row -- carrying the calibration's own uncertainty into every future probabilistic run instead of a single point estimate."),
    Spacer(1, 10),

    p("PSA Setup", "ManualHeading"),
    p("Probabilistic Sensitivity Analysis draws each configured input from a distribution on every iteration instead of holding it at its base value. PSA Parameters is the table of inputs varied this way: each row targets either a Parameter/Variable by name, or one shape/scale slot of a Distribution, and carries its own PSA distribution type (Beta, Normal, Gamma, ...) and values (for example mean, se)."),
    p("Correlation", "ManualHeading"),
    p("Two PSA Parameters rows can be linked with a pairwise correlation coefficient so they no longer vary independently across iterations. This is for a simple two-way relationship; for three or more parameters that must move together under a shared structure, use Correlated Distribution instead."),
    p("Correlated Distribution", "ManualHeading"),
    p("A Correlated Distribution group draws several PSA Parameters rows jointly from a single named distribution, instead of independently. Two kinds exist in DECIDE: Dirichlet and Multivariate Normal. A decision tree has no Markov transitions, so only Multivariate Normal applies here -- Dirichlet groups a Markov state's outgoing transitions and is covered in Part C."),
    p("Pick two or more existing PSA Parameters rows, set each one's mean and standard deviation, and fill in the pairwise correlation between every pair of members. An inconsistent correlation matrix is automatically repaired to the nearest valid one before sampling."),
    p("Applying a group generates one set of jointly-drawn values per PSA iteration and stores them against the group; the group panel shows Applied / Not applied so it is clear when a change still needs to be applied before the next PSA run picks it up. A group only affects the PSA iterations that come after it is applied -- the model's own base-case values are unchanged."),
    Spacer(1, 10),

    PageBreak(),
    p("Value of Information (EVPI/EVPPI)", "ManualHeading"),
    p("The PSA results panel (shared across the decision tree and Markov model types) reports the Expected Value of Information alongside the CE plane, CEAC/CEAF, and convergence diagnostics — the standard HTA/payer measure of how much uncertainty is costing the decision, and whether further research is worth funding."),
    table(["Metric", "What it shows"], [
        ["EVPI", "The per-patient value of resolving all PSA uncertainty at once, plotted across a grid of willingness-to-pay thresholds centred on the model's own WTP setting."],
        ["EVPPI by parameter", "EVPI broken down by PSA parameter via polynomial regression of each strategy's net monetary benefit against that parameter's sampled values (the Strong/Oakley method). The regression degree is chosen per parameter by cross-validation and shown in the table's Degree column."],
        ["EVPPI for a group", "Select 2+ parameters to estimate their joint EVPPI — the value of a study resolving all of them together, not generally the sum of their individual EVPPI, since parameters can interact."],
    ]),
    Spacer(1, 10),
    p("Population EVPI/EVPPI", "ManualHeading"),
    p("EVPI/EVPPI are per-patient by default. The Population inputs — annual eligible population, time horizon, discount rate — scale every value up to the total population the decision affects, using the same discounting convention the model's own cost calculations use: effective population = the sum, over each year of the horizon, of annual population / (1 + discount rate)^year, with the first year undiscounted. This is the number comparable to the cost of a proposed further study. Leaving the annual population at 0 shows only the per-patient values."),
    PageBreak(),

    # =====================================================================
    # PART C -- Markov Model
    # =====================================================================
    p("Part C -- Markov Model", "ManualPart"),

    p("Engine variables", "ManualHeading"),
    table(["Variable", "Meaning"], [
        ["_cycle", "The model cycle, starting at 0. A decision tree evaluates once with _cycle = 0. In a Markov model, _cycle increases by one every simulated cycle."],
        ["_tenure", "The number of consecutive cycles spent in the current Markov state, starting at 1. Decision trees evaluate with _tenure = 1."],
    ]),
    Spacer(1, 10),
    p("Markov tenure states", "ManualHeading"),
    p("Enable Tenure applied on a Markov state and set No. of tenure to model time spent in that state. DECIDE represents the state internally as temporary tunnel stages: entry starts at _tenure=1, each self-loop advances to the next tenure stage, and the final stage retains the self-loop so _tenure stays at its configured maximum. All transitions to another state are available from every tenure stage; a transition to another state restarts that destination state at _tenure=1."),
    p("Costs on a tenure-enabled state: Initial cost applies at _tenure=1 to everyone in the state in that cycle, including the portion that leaves during that same cycle (that portion receives only Initial cost, no Terminal cost); Recurring cost applies from _tenure=2 to the cohort portion remaining in the state; Terminal cost applies from _tenure=2 to the portion leaving the state next cycle (replacing recurring cost for that portion); at the time horizon, a cohort that remains receives recurring cost rather than terminal cost. To reproduce a plain per-cycle state cost c (as in TreeAge state rewards), enter c as Initial, Recurring and Terminal cost."),
    Spacer(1, 10),

    p("Markov Editor", "ManualHeading"),
    p("<i>Pending — the canvas mechanics for Markov (State and Transition editing, Strategies, absorbing states, tenure settings) have not yet been verified against the current app UI for this version of the manual and will be filled in a follow-up pass. See Engine variables above for tenure-enabled states, and Part B → File panel / Decision Tree Editor for the mechanics shared with the tree editor.</i>"),
    Spacer(1, 10),

    p("File panel", "ManualHeading"),
    p("Same as Part B → File panel."),

    p("Model Calibration", "ManualHeading"),
    p("Same mechanism as Part B → Model Calibration. The one difference is the shape of a target row: for a Markov model each row is a (Cycle, State, Value, Weight) point along the model's cohort trace, and Applying a result additionally shows a chart overlaying the calibrated model's trace against the target points — both already described in Part B."),

    p("PSA Setup", "ManualHeading"),
    p("PSA Parameters and Correlation work exactly as in Part B → PSA Setup. Correlated Distribution's Dirichlet kind is available here, since a Markov model's states have outgoing transitions to group:"),
    table(["Kind", "Use for / draws"], [
        ["Dirichlet", "A Markov state's outgoing transition probabilities, so they vary independently but always sum to exactly 1 on every PSA iteration. One row per outgoing transition, each a bare Parameter reference."],
        ["Multivariate Normal", "Any two or more PSA Parameters rows that should move together under a general correlation structure (see Part B for detail)."],
    ]),
    Spacer(1, 10),
    p("Dirichlet: pick the Markov state; DECIDE finds its outgoing transitions automatically and requires every one of them to be a plain Parameter or Variable reference. A formula, or a transition using a complement / 'remaining probability' marker, cannot be part of a Dirichlet group, since the group's own draws already sum to 1 and the complement would then always resolve to 0. Set a concentration value (alpha) for each outgoing transition; alpha's relative sizes set the group's average split, and its overall scale sets how tightly the draws cluster around that split. If a transition is not yet a PSA Parameters row, DECIDE offers to add it with a starting Beta distribution before the group can be applied."),
    p("A Markov model's Dirichlet grouping applies to the currently open Strategy's transitions; it does not automatically extend to a state of the same name in a different Strategy."),
    p("Only Dirichlet and Multivariate Normal joint sampling are available for PSA today. A Multinomial-style joint distribution over discrete outcome counts is not yet implemented; it is a planned addition for a future model type where the outcomes it would apply to are counts of simulated events rather than continuous Markov transition probabilities."),
    Spacer(1, 10),

    p("Strategy comparison and ICER", "ManualHeading"),
    p("Base case results list the strategies from the cheapest to the most expensive. The cheapest is the reference. Each of the others is compared with the <b>previous strategy that is not dominated</b>, not simply with the row above it."),
    p("<b>Dominated</b>: a strategy that costs more than a cheaper one but adds no effect. It is shown as <i>Dominated</i> and is never used as the comparator of the next strategy. <b>Extendedly dominated</b>: a strategy whose ICER against its comparator is higher than the ICER of the next, more effective strategy; it is shown as <i>Extendedly dominated</i>. Incremental cost and effect are measured against that comparator, and the ICER is their ratio. The same table is used for Decision Tree, Markov and Microsimulation models."),
    p("Value of Information (EVPI/EVPPI)", "ManualHeading"),
    p("Same as Part B → Value of Information (EVPI/EVPPI). The PSA results panel and its EVPI/EVPPI reporting are shared unchanged between decision tree and Markov models."),
]


# =====================================================================
# PART D -- Partitioned Survival Model. Rendered from the Part D section of DECIDE_manual.md
# (single source of truth: edit the markdown, rebuild the PDF).
# =====================================================================
import re as _re


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def _md_table(rows: list[list[str]]) -> Table:
    n = len(rows[0])
    widths = [1.65 * inch, 5.55 * inch] if n == 2 else [1.5 * inch, 2.0 * inch, 3.7 * inch]
    data = [[Paragraph(f"<font color='white'><b>{_inline(c)}</b></font>", styles["ManualBody"]) for c in rows[0]]]
    data += [[Paragraph(_inline(c), styles["ManualBody"]) for c in r] for r in rows[1:]]
    t = Table(data, repeatRows=1, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BLUE), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7d0f4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _md_section_flowables(md_text: str, start_heading: str) -> list:
    lines = md_text.splitlines()
    begin = next(i for i, l in enumerate(lines) if l.strip() == start_heading)
    end = len(lines)
    for i in range(begin + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    flow: list = []
    i = begin
    base = Path(__file__).parent
    while i < end:
        line = lines[i]
        if line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:].replace(" -- ", " -- ")), styles["ManualPart"]))
        elif line.startswith("### ") or line.startswith("#### "):
            flow.append(Paragraph(_inline(line.lstrip("# ")), styles["ManualHeading"]))
        elif line.startswith("|"):
            rows = []
            while i < end and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("- ") for c in cells):
                    rows.append(cells)
                i += 1
            flow += [_md_table(rows), Spacer(1, 10)]
            continue
        elif line.startswith("!["):
            m = _re.match(r"!\[(.*?)\]\((.*?)\)", line)
            img = base / m.group(2)
            if img.exists():
                from reportlab.lib.utils import ImageReader
                w, h = ImageReader(str(img)).getSize()
                width = 6.6 * inch
                flow += [Image(str(img), width=width, height=width * h / w),
                         Paragraph(f"<i>{_inline(m.group(1))}</i>", styles["ManualBody"]), Spacer(1, 6)]
        elif _re.match(r"^\d+\. ", line) or line.startswith("- "):
            bullet = "&bull; " if line.startswith("- ") else ""
            flow.append(Paragraph(bullet + _inline(line[2:] if line.startswith("- ") else line),
                                  ParagraphStyle(name="li", parent=styles["ManualBody"], leftIndent=14, spaceAfter=4)))
        elif line.strip():
            flow.append(Paragraph(_inline(line), styles["ManualBody"]))
        i += 1
    return flow


story.append(PageBreak())
story += _md_section_flowables((Path(__file__).parent / "DECIDE_manual.md").read_text(encoding="utf-8"),
                               "## Part D -- Partitioned Survival Model (PSM)")

story.append(PageBreak())
story += _md_section_flowables((Path(__file__).parent / "DECIDE_manual.md").read_text(encoding="utf-8"),
                               "## Part E -- Microsimulation")



doc = ManualDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    rightMargin=0.55 * inch,
    leftMargin=0.55 * inch,
    topMargin=0.55 * inch,
    bottomMargin=0.65 * inch,
    title="DECIDE Manual",
    author="DECIDE",
)
doc.multiBuild(story, onFirstPage=page_number, onLaterPages=page_number)
print(OUTPUT)
