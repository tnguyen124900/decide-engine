# DECIDE Manual

*Version 1.2 -- 1 October 2026*

## Part A -- The DECIDE Expression Language

The knowledge in this part is shared by every model type in DECIDE (Decision Tree, Markov, and future model types): the same expression evaluator, the same functions, and the same Parameters/Variables/Distributions/Tables data model underlie every Cost, Effect, Probability, and definition field in the app.

### Expression engine

DECIDE expressions are evaluated with Python-compatible semantics. The same expression rules are available in decision trees and Markov models. Parameters are numeric inputs; variables are named expressions that may reference parameters and other variables.

Example:

```text
Parameter: rate_AE_3yr = 0.5
Variable: prob_AE = RateToProb(rate_AE_3yr, 1, 1/4)
```

The resulting probability is approximately `0.1175030974`.

### Operators

| Operator | Meaning |
| --- | --- |
| `+` | Addition |
| `-` | Subtraction or negation |
| `*` | Multiplication |
| `/` | Division |
| `**` | Exponentiation |
| `%` | Modulus |
| `//` | Floor division |
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |
| `and` | Logical AND |
| `or` | Logical OR |
| `not` | Logical NOT |
| `( ... )` | Grouping and precedence |

Expressions follow Python precedence. Use parentheses when the intended order is not obvious.

### Constants

- `e` is Euler's number.
- `pi` is the mathematical constant pi.
- Trigonometric functions use radians.

### Scalar functions

| DECIDE function | Meaning |
| --- | --- |
| `abs(x)` | Absolute value |
| `arccos(x)` | Inverse cosine, in radians |
| `arcsin(x)` | Inverse sine, in radians |
| `arctan(x)` | Inverse tangent, in radians |
| `bound(x, (a,b))` or `bound(x, [a,b])` | Clamp `x` into `[a, b]` |
| `cbrt(x)` | Cube root |
| `ceil(x)` | Ceiling |
| `comb(n, k)` | Binomial coefficient; `n >= k >= 0` |
| `cos(x)` | Cosine |
| `cosh(x)` | Hyperbolic cosine |
| `erf(x)` | Error function |
| `exp(x)` | Exponential, `e**x` |
| `factorial(n)` | Factorial for a non-negative integer |
| `floor(x)` | Floor |
| `gamma(x)` | Gamma function |
| `hypot(x, y)` | Hypotenuse, `sqrt(x**2 + y**2)` |
| `if(condition, a, b)` | Return `a` when true, otherwise `b` |
| `invErf(x)` | Inverse error function |
| `lgamma(x)` | Natural logarithm of the absolute Gamma function |
| `ln(x)` | Natural logarithm |
| `log(x, b)` | Logarithm of `x` to base `b`; `log(x, e)` equals `ln(x)` |
| `log10(x)` | Base-10 logarithm |
| `logit(p)` | Log odds, `ln(p / (1-p))` |
| `logis(x)` | Logistic function, `1 / (1 + exp(-x))` |
| `max(a, b)` | Maximum; accepts multiple values |
| `min(a, b)` | Minimum; accepts multiple values |
| `round(x)` | Python-compatible rounding to zero decimal places |
| `round(x, n)` | Python-compatible rounding to `n` decimal places |
| `sign(x)` | `-1`, `0`, or `1` according to the sign of `x` |
| `sin(x)` | Sine |
| `sinh(x)` | Hyperbolic sine |
| `sqrt(x)` | Square root |
| `tan(x)` | Tangent |
| `tanh(x)` | Hyperbolic tangent |

### Probability and rate functions

| DECIDE function | Meaning |
| --- | --- |
| `ProbToProb(p, t1, t2)` | Convert probability `p` from interval `t1` to interval `t2` |
| `ProbToRate(p, t1, t2)` | Convert probability to a rate |
| `RateToProb(r, t1, t2)` | Convert rate to a probability |

The conversion formulas are:

```text
ProbToProb(p, t1, t2) = 1 - (1-p)**(t2/t1)
ProbToRate(p, t1, t2) = -ln(1-p) * t2/t1
RateToProb(r, t1, t2) = 1 - exp(-r * t2/t1)
```

### Distribution functions

The `distribution` argument is the name of a distribution defined in the Distribution section. Its type and parameters are not repeated in the function call.

| DECIDE function | Meaning |
| --- | --- |
| `DistMean(distribution)` | Mean of the configured distribution |
| `DistSample(distribution)` | Random sample from the configured distribution |
| `DistCDF(distribution, s)` | Cumulative probability, $P(X <= s)$ |
| `DistPPF(distribution, p)` | Value $s$ whose cumulative probability is `p` |
| `DistSampleCond(distribution, interval)` | Random sample conditional on an interval such as `[lower, upper]` or `(lower, upper)` |
| `DistToProb(distribution, t1, t2)` | Conditional transition probability during `[t1, t1+t2]` given no event before `t1` |
| `DistToTime(distribution, t1)` | Random remaining time until the event for someone still event-free at `t1` (`t1` optional, default 0). Measured from `t1`, like `t2` in `DistToProb` |
| `DistTrim(distribution, [min, max])` | Random sample trimmed to `min <= X <= max` |
| `DistSurv(distribution, s)` | Survival probability, $1-DistCDF(distribution,s)$ |
| `DistHazard(distribution, s)` | Hazard, `PDF(s) / survival(s)` |

`DistToProb` is not a CDF or PDF. It is calculated from the **conditional survival** $S(t_1+t_2)/S(t_1)$, where $S(t) = 1-F(t)$ is the survival function:

$$
P(t1 < T <= t1+t2 | T > t1) =
\frac{F(t1+t2)-F(t1)}{1-F(t1)} = 1 - \frac{S(t1+t2)}{S(t1)}
$$

**One formula, two functions.** DECIDE implements the conditional survival $S(t_1+t_2)/S(t_1)$ once and exposes it as two functions that answer different questions:

| Function | Question | Output |
| --- | --- | --- |
| `DistToProb(distribution, t1, t2)` | What share of those still event-free at `t1` have the event during the next `t2`? | A probability, e.g. a per-cycle transition probability in a Markov model |
| `DistToTime(distribution, t1)` | When does *this* individual, event-free at `t1`, have the event? | A random time $d \ge 0$ from `t1`, solving $S(t_1+d)/S(t_1) = u$ for a random $u$ in (0, 1] |

Being the same formula, they always agree: the share of `DistToTime(distribution, t1)` draws that are $\le t_2$ equals `DistToProb(distribution, t1, t2)`. Example: for a Distribution `prog` of type Weibull with shape 1.3 and scale 6, `DistToProb(prog, 2, 1)` = $1 - S(3)/S(2)$ ≈ 0.153, and about 15.3% of `DistToTime(prog, 2)` draws are at most 1.

The distribution's own parameters (e.g. Weibull shape 1.3, scale 6) are set in the Distribution section, not in the function call.

Example:

```text
Distribution name: survival_dist
Type: Weibull
shape = 1.5
scale = 8

DistToProb(survival_dist, 2, 0.25)
```

For Python runtime support, DECIDE maps configured distributions to `scipy.stats` objects: `.mean()`, `.rvs()`, `.cdf()`, `.ppf()`, `.sf()`, and `.pdf()`.

### Distribution types and parameters

**DECIDE follows the parameterization of the R package flexsurv** (which for most families is also the parameterization of R's own `d`/`p`/`q`/`r` functions, such as `pweibull` and `qgamma`). Values fitted with flexsurv, or reported in a paper that uses flexsurv, can be typed straight into the Distribution table. The Gamma distribution uses **shape and rate**, as flexsurv does (scale = 1/rate).

Time-to-event distributions, with `t` in the distribution's own time unit:

| Distribution | Parameters | Survival function S(t) and notes |
| --- | --- | --- |
| Exponential | `rate` | $S(t) = e^{-rate \cdot t}$; mean $1/rate$ |
| Weibull | `shape`, `scale` | $S(t) = \exp(-(t/scale)^{shape})$; mean $scale \cdot \Gamma(1 + 1/shape)$ |
| Gamma | `shape`, `rate` | mean $shape/rate$, variance $shape/rate^2$; scale = 1/rate |
| Log-normal | `meanlog`, `sdlog` | $S(t) = 1 - \Phi((\ln t - meanlog)/sdlog)$ |
| Log-logistic | `shape`, `scale` | $S(t) = 1 / (1 + (t/scale)^{shape})$; median = `scale` |
| Gompertz | `shape`, `rate` | hazard $h(t) = rate \cdot e^{shape \cdot t}$; $S(t) = \exp(-(rate/shape)(e^{shape \cdot t} - 1))$ |
| Generalized gamma | `mu`, `sigma`, `Q` | Prentice (flexsurv `gengamma`) parameterization; Q = 0 is the log-normal |

**Time unit.** DECIDE never converts time units. The distribution's parameters must be in the same unit as the arguments of `DistToProb`, `DistToTime`, `DistSurv` and so on, and as the model cycle (for example, a Weibull fitted in years used with monthly cycles needs `DistToProb(d, t1/12, 1/12)`).

**Entering values reported in other conventions.** The same name can mean different parameters in different software, so convert before typing:

| Source | Reported as | Enter in DECIDE |
| --- | --- | --- |
| R `pweibull`, `flexsurvreg(dist = "weibull")` | shape, scale | as reported |
| `flexsurvreg(dist = "weibullPH")`, Stata `streg, dist(weibull)` (proportional hazards) | cumulative hazard $\lambda t^{\gamma}$ | `shape` = $\gamma$, `scale` = $\lambda^{-1/\gamma}$ |
| Paper or software with cumulative hazard $(\lambda t)^{\gamma}$ (for example the DARTH time-dependent Sick-Sicker tutorial) | $\lambda$, $\gamma$ | `shape` = $\gamma$, `scale` = $1/\lambda$ |
| R `survival::survreg` Weibull (accelerated failure time) | Intercept, Log(scale) | `scale` = $\exp(\text{Intercept})$, `shape` = $1 / \text{survreg scale}$ |
| `survreg` log-logistic | Intercept, scale | `scale` = $\exp(\text{Intercept})$, `shape` = $1 / \text{survreg scale}$ |
| `survreg` log-normal | Intercept, scale | `meanlog` = Intercept, `sdlog` = scale |
| Exponential given as a mean time $\theta$ | $\theta$ | `rate` = $1/\theta$ |
| Gamma given as shape and scale $\theta$ | $k$, $\theta$ | `shape` = $k$, `rate` = $1/\theta$ |
| Gamma given as mean $m$ and standard deviation $s$ | $m$, $s$ | `shape` = $(m/s)^2$, `rate` = $m/s^2$ |
| `flexsurvreg(dist = "gengamma.orig")` | shape $b$, scale $a$, $k$ | `Q` = $1/\sqrt{k}$, `sigma` = $1/(b\sqrt{k})$, `mu` = $\ln a + \ln k / b$ |

With flexsurv output, use the natural-scale estimates (`est` in `res`), not the log-scale ones (`res.t`), and with covariates compute the parameter at the covariate values of the group being modelled before typing it.

The Weibull is the most common source of confusion: a shape of 1.1 with $\lambda$ = 0.08 gives `scale` = 12.5 when the cumulative hazard is $(\lambda t)^{\gamma}$ but `scale` = 9.94 when it is $\lambda t^{\gamma}$, and the one-cycle event probability at time 0 is 0.060 versus 0.077.

**Gamma and older models.** Earlier versions of DECIDE stored the Gamma distribution as `k` (shape) and `theta` (scale). Such models are converted when the Distribution row is loaded: `k` becomes `shape` and `theta` becomes `rate = 1/theta` (an expression is wrapped as `1/(expression)`), and the distribution, and every result that depends on it, is unchanged. A PSA row that targets a Gamma slot by its old name `theta` must be pointed at `rate` again.

Other distributions, with the parameters shown in the Distribution table:

| Distribution | Parameters | Definition |
| --- | --- | --- |
| Beta | `a`, `b` | R `shape1`, `shape2`; mean $a/(a+b)$ |
| Normal | `mu`, `sigma` | standard deviation, not variance |
| Truncated Normal | `mu`, `sigma`, `a`, `b` | `mu` and `sigma` describe the untruncated normal; `a`, `b` are the bounds |
| Uniform | `a`, `b` | minimum and maximum |
| Triangular | `a`, `b`, `c` | minimum, mode, maximum |
| PERT | `a`, `b`, `c` | minimum, most likely, maximum; Beta with weight 4 |
| Poisson | `lambda` | mean count |
| Binomial | `n`, `p` | number of trials, probability |
| Bernoulli | `p` | probability of success |
| Geometric | `p` | number of trials until the first success (starts at 1) |
| Negative Binomial | `r`, `p` | number of failures before the `r`-th success |
| Discrete Uniform | `a`, `b` | integers from `a` to `b`, both included |
| Hypergeometric | `w`, `b`, `n` | white, black, number drawn |
| Categorical | `p` | list of probabilities that sums to 1 |
| Student's t | `nu` | standard t (location 0, scale 1) |
| Chi-Square | `k` | degrees of freedom |
| Cauchy | `x0`, `gamma` | location, scale |
| Half-Cauchy | `gamma` | scale |
| Half-Normal | `sigma` | scale |
| Laplace | `mu`, `b` | location, scale |
| Logistic | `mu`, `s` | location, scale |
| Gumbel | `mu`, `beta` | maximum-extreme (right-skewed) form |
| Pareto | `alpha`, `k` | shape, minimum (type I) |
| Zipf | `s`, `n` | exponent, number of elements |

A life-table column is a probability or a rate depending on its label: `qx` is an annual probability and is used directly, `mx` is a rate and goes through `RateToProb`.

### Tables and Lookup

A Table is a named grid of reference data -- age-banded mortality, a cost list, a
survival/hazard schedule -- entered directly or uploaded from a CSV or Excel
file. Columns come from the file's header row, or, when **Headers not
included** is enabled, from generated names `column_1`, `column_2`, and so
on. Every column can also be addressed by a 0-based index: the first column
is index `0`, the second is index `1`, and so on.

Tables are available in both Decision Tree and Markov models, and a table's
own values are looked up with the same syntax in either engine.

#### Lookup syntax

```text
table_name[lookup_value]
table_name[lookup_value, result_column]
table_name[lookup_value, lookup_column, result_column]
table_name[lookup_value, lookup_column, result_column, method]
```

`lookup_value` is any DECIDE expression -- a parameter, a variable, `_cycle`,
or a formula -- evaluated and matched against the table's key column.
`lookup_column` and `result_column` each accept a quoted column name (for
example `"Age"`) or a 0-based column index (for example `0`). `method` is a
quoted interpolation method: `"exact"`, `"truncate"`, `"linear"`, or
`"spline"`; when omitted it defaults to `"linear"`.

| Form | lookup column | result column |
| --- | --- | --- |
| `table_name[lookup_value]` | column `0` | the table's **Default result column** setting, or the sole other column when the table has exactly 2 columns |
| `table_name[lookup_value, result_column]` | column `0` | `result_column` |
| `table_name[lookup_value, lookup_column, result_column]` | `lookup_column` | `result_column` |
| `table_name[lookup_value, lookup_column, result_column, method]` | `lookup_column` | `result_column` |

A table with more than two columns needs a **Default result column**
configured in the Tables editor before `table_name[lookup_value]` can be
used; otherwise, give the result column explicitly.

#### Interpolation methods

| Method | Behaviour |
| --- | --- |
| `"exact"` | Requires a row whose key matches `lookup_value` exactly; raises an error (listing the available keys) if there is no match. |
| `"truncate"` | Uses the row with the highest key at or below `lookup_value` (a step function). Below the lowest key, the lowest row is used. |
| `"linear"` (default) | Linearly interpolates between the two rows that bracket `lookup_value`. |
| `"spline"` | Fits a cubic spline through the table's rows (requires at least 3 rows with unique keys) and evaluates it at `lookup_value`. |

For `"linear"` and `"spline"`, a `lookup_value` outside the table's range is
clamped to the nearest edge row rather than extrapolated.

#### Examples

Age-banded mortality, linear interpolation between the surrounding rows:

```text
Table: t_mortality, columns Age | Mortality

t_mortality[Current_age, "Age", "Mortality"]
```

A Markov transition probability that changes every cycle, read straight from
a table keyed on `_cycle`:

```text
t_mortality[_cycle, "Cycle", "Prob"]
```

An exact age-band cost lookup, using column indices instead of names:

```text
t_cost[Current_age, 0, 1, "exact"]
```

Combining Lookup with `RateToProb` reproduces a hazard/survival table
function without a dedicated one: look up the rate for the current cycle,
then convert it to a cycle probability the same way any other rate is
converted:

```text
RateToProb(t_hazard[_cycle, "Cycle", "Rate"], 1, 1)
```

#### Notes

- Table values are numeric. A cell that cannot be read as a number is
	skipped, and a row missing either the lookup or result column's value is
	skipped as well.
- An unknown table name, an unknown column, or an unrecognised method raises
	a clear error naming the problem, so a broken Lookup formula fails loudly
	rather than silently returning a wrong value.

#### TableToProb and TableToTime

The table counterparts of `DistToProb`/`DistToTime`, for survival or mortality **tables** (e.g. a national life table) instead of parametric distributions. They use the same conditional-survival formula, with $S(x) = 1 - F(x)$ built from the table.

| Function | Returns |
| --- | --- |
| `TableToProb(table, t1, t2[, value_column[, type]])` | Probability of the event in `(t1, t1+t2]` for someone event-free at `t1` |
| `TableToTime(table, t1[, value_column[, type]])` | Random remaining time from `t1` until the event for someone event-free at `t1` |

- The table's **first column** is the index (e.g. `age`); `t1` is on that scale.
- `value_column` (quoted, e.g. `"qx"`) defaults to the table's default value column, else its second column.
- `type` (quoted) says how values are read: `"conditional"` (default -- a life table's q(x): probability of the event in [x, next x) given event-free at x), `"probability"` (per-row probability, rows sum to 1), or `"cumulative"` (cumulative probability, last row 1).
- Values are interpolated within each interval (events spread uniformly within each year).
- The table name must be a plain name (letters, digits, `_`; e.g. `life_table`).

They agree by construction: the share of `TableToTime(table, t1)` draws that are $\le t_2$ equals `TableToProb(table, t1, t2)`. For a q(x) table at a whole age, `TableToProb(life_table, 60, 1)` is exactly q(60), and `TableToProb(life_table, 60, 2)` = $1-(1-q_{60})(1-q_{61})$.

Example:

```text
Markov background mortality per cycle:   TableToProb(life_table, startAge + _cycle, 1)
```

### Summary functions

| DECIDE function | Meaning |
| --- | --- |
| `mean(x1, x2, ..., xn)` | Arithmetic mean |
| `prod(x1, x2, ..., xn)` | Product |
| `quantile({x1, x2, ..., xn}, q)` | Linear-interpolated quantile, `0 <= q <= 1` |
| `sd(x1, x2, ..., xn)` | Sample standard deviation, denominator `n-1` |
| `sum(x1, x2, ..., xn)` | Sum |
| `var(x1, x2, ..., xn)` | Sample variance, denominator `n-1` |

Quantile examples:

- Q1: `quantile({x1, x2, ..., xn}, 0.25)`
- Q2 / median: `quantile({x1, x2, ..., xn}, 0.50)`
- Q3: `quantile({x1, x2, ..., xn}, 0.75)`

### Parameters vs Variables

- **Parameter**: a root input, with `name` / `base` / `low` / `high` / `source`. `base` can be a number or a formula referencing other Parameters. `low`/`high` feed Sensitivity Analysis (One-way/Tornado) and Model Calibration bounds; they are optional otherwise. A Parameter can have a PSA distribution attached (see PSA Setup in Part B and Part C).
- **Variable**: a quantity computed from other Parameters/Variables via a `definition` (formula). In a Markov model, a Variable's definition can depend on `_cycle`/`_tenure` (see Engine variables in Part C), so a Markov Variable's value can change by cycle; a Decision Tree Variable always evaluates with `_cycle = 0`, `_tenure = 1`, so its value is fixed.
- Parameters and Variables share one namespace -- a Variable's name cannot duplicate a Parameter's name.
- **Renaming is automatic.** Rename a Parameter, Variable, Distribution, Table or Tracker in its input table and every place that uses the old name is rewritten to the new one -- canvas formulas (branches, states, transitions), other Variables' definitions, Distribution parameters, trackers and rates, PSA setup and Sensitivity/Calibration selections -- in Decision Tree, Markov, PSM, DES, Microsimulation and ABM alike. Text inside quotes (a table column such as `tbl["age"]`) and longer names that merely contain the old one are never touched. If the new name is already taken, nothing is rewritten and the duplicate-name warning shows.

Use a parameter for a direct numeric input, and a variable for a derived value:

```text
rate_AE_3yr = 0.5
prob_AE = RateToProb(rate_AE_3yr, 1, 1/4)
```

Then use `prob_AE` in a decision-tree branch probability, terminal cost/effect, Markov initial probability, state cost/effect, or transition probability.

## Part B -- Decision Tree

### File panel

Decision Tree, Markov, PSM, ABM and DynaTrans share the same File panel; NMA has Save / Save as / Open with its own autosave.

- **Save:** writes the model straight to Folder/Filename, overwriting without asking; once a file is saved or opened, Save keeps updating it.
- **Save as:** downloads a copy through the browser.
- **Folder / Saved models / Open saved model:** choose the folder (remembered; each model type defaults to its own subfolder of models/) and open a model saved there.
- **Autosave and Restore:** every change is autosaved (five rotating backups); reopening the app offers a Restore banner for unsaved earlier work. There is no unsaved-changes warning.

### Decision Tree Editor

The Tree Editor draws the tree on a canvas. Branch labels, probabilities, costs and effects are edited on the canvas itself; node names are not shown. Strategies are the branches leaving the Decision node.

- **Strategy branches:** a Decision-node branch only carries a label -- no probability, no $ and no E input. Cost and effect belong to the chance branches and terminals below it.
- **Path results on the canvas:** beside each terminal, the path's probability p, total cost and total effect (branch increments plus the terminal value, the terminal discounted by its year). Toggle with "Show path results".
- **Strategy totals:** under each strategy node, the base-case expected totals labelled `total_` plus the outcome symbol (e.g. `total_$`, `total_E`); they equal the sum of p x value over the strategy's paths.
- **Base case results:** a Paths table (path shown as Strategy -> label -> label) with undiscounted and discounted cost/effect, a Paths CSV and a Trace CSV of the cumulative steps, and a warning if paths do not reconcile with the engine totals.
- **Cost-only models:** delete the Effect row in Outcomes to run cost-only; effect inputs are zeroed and restored if an Effect row is added again.
- **Text style:** font, size and B / I / U toolbar, set separately per text type (not synced).
- **Save / Save as / Open:** Save writes to Folder/Filename and overwrites without asking; Save as downloads a copy; Open loads from the Saved models list.
- **Autosave and Restore:** the model is autosaved; reopening the app offers a Restore banner for unsaved earlier work.

### Model Calibration

Model Calibration adjusts a model's own Parameters and Variables so that its output matches data observed in the real world -- a known outcome rate, or, for Markov models, a Kaplan-Meier-style curve of state occupancy over time. It works the same way across model types: a "Model calibration" button in the sidebar opens a panel below the model's Parameters/Variables/Distributions/Tables cards, with the same overall flow in every case -- pick which inputs may move, enter the data they should be matched against, choose an algorithm, run it, then apply or discard the result. The Markov-specific target-row shape is covered in Part C; everything else on this page applies to both model types identically.

#### Target data

A decision tree has no cycles, so each target row is a single observed number for one strategy; a Markov model is calibrated against a full trace, so each target row is one (cycle, state) pair. The target table is edited directly in the panel and rows can be added or removed freely, the same way the Tables section works elsewhere in DECIDE.

| Model | Target row columns | Notes |
| --- | --- | --- |
| Markov | Cycle, State, Value, Weight | State is picked from the states of the strategy being calibrated. Value is the observed occupancy fraction for that state at that cycle -- for example, digitised points off a published Kaplan-Meier curve. |
| Decision tree | Metric, Target ID, Value, Weight | Metric is one of `total_cost`, `total_effect`, or `terminal_probability`. Target ID names a terminal node and only applies to `terminal_probability`; it is ignored for the other two metrics. |

Weight lets one target row count for more than another when several rows are being matched at once (for example, to emphasise a later, more clinically important cycle over an early one) -- it defaults to 1 and rarely needs changing. A `total_cost`/`total_effect` target compares against the same strategy-level rollup shown in the model's own Base Case results; a `terminal_probability` target compares against the probability of reaching one specific terminal node, computed as the product of the branch probabilities along the single path from the strategy's root to that node.

#### Selecting what to calibrate

The panel lists every Parameter and Variable available to the model (the same list the Tornado sensitivity analysis draws from), each with a checkbox and its current base/low/high values. Checking an input makes it eligible to move during calibration; every checked input must already have both a Low and a High bound set (in the Parameters/Variables section above) and High must be greater than Low, or the Run button stays disabled with an explanation of what is missing. Calibration only ever adjusts the inputs that are checked -- everything else in the model stays fixed at its current value for the run.

#### Algorithms

Five algorithms are available, in two families. Local Optimization, Random Search, and Grid Search each return one best-fit point per input -- a single calibrated value to adopt. Bayesian (SIR) and Bayesian (MCMC) instead return a posterior distribution per input, reflecting how much the target data actually constrains that input rather than collapsing it to one number.

| Algorithm | How it works | Best for | Trade-off |
| --- | --- | --- | --- |
| Local Optimization | Starts from each input's current base value and takes a sequence of steps in the direction that reduces the mismatch with the target data fastest, stopping once it converges. | The common case, where the mismatch changes smoothly as inputs move. Fastest of the five, and the cheapest to run. | Can settle in a local best-fit that is not the true best fit if the starting point is far off or several inputs interact. |
| Random Search | Draws a set number of random candidate combinations of the checked inputs (within their Low/High bounds) and keeps the one with the smallest mismatch, optionally refining that point afterwards with Local Optimization. | A first, coarse pass before refining, or any time the mismatch is noisy rather than smooth. | Slower than Local Optimization for the same precision, and only as good as the best point it happened to try. |
| Grid Search | Same idea as Random Search, but the candidates are an evenly spaced grid over the checked inputs' bounds instead of random draws. | Low-dimensional problems (one or two calibrated inputs) where covering the space evenly matters more than covering it densely. | Cost grows fast with the number of calibrated inputs -- a grid with, say, 5 points per dimension needs 25 evaluations for 2 inputs but 125 for 3. |
| Bayesian (SIR) | Sampling-Importance-Resampling: draws a large batch of candidates from each input's prior (its existing PSA distribution when one is configured, otherwise a Uniform distribution over its Low/High bounds), scores each by how well it matches the target data, and resamples from that batch in proportion to fit to produce the posterior. | A single, cheap-to-evaluate pass at a genuine posterior, and the more robust of the two Bayesian options when the prior is a reasonable starting guess. | Needs enough prior draws to cover the region the posterior actually lives in -- too few, and the reported credible interval can come out wider or more jagged than the truth. |
| Bayesian (MCMC) | Metropolis-Hastings: walks a chain of candidates through parameter space, each step randomly proposed near the last and accepted or rejected by how much it improves or worsens the fit, so that after enough steps the chain's visited points approximate the posterior. | Cases where SIR's one-shot sampling struggles -- a narrow posterior relative to the prior, or several calibrated inputs at once. | More settings to tune (chain length, burn-in, step size) and, step for step, no cheaper than SIR -- the chain also needs a short pilot run first to calibrate its own noise assumption, adding a small number of extra evaluations. |

Local Optimization, Random Search, and Grid Search share their settings and results layout. Random Search exposes a Samples count and an optional "Refine with Local Optimization afterwards" checkbox; Grid Search exposes Grid points per parameter (per dimension) and the same refine option. Both read their search bounds from each checked input's Low/High.

#### Bayesian Calibration (SIR and MCMC)

Both Bayesian algorithms score candidates against the same target data as Local/Random/Grid Search, but instead of narrowing to one point they report, for every calibrated input: a posterior mean and standard deviation, and a 95% credible interval. Each candidate's prior draw reuses that input's existing PSA distribution when one is already configured in PSA Parameters, and falls back to a Uniform distribution over its Low/High bounds otherwise -- the same implicit prior Random and Grid Search already use for their own candidate draws.

| Setting | Applies to | Meaning |
| --- | --- | --- |
| Prior draws | SIR | How many candidate parameter sets to draw from the prior and score. More draws give a smoother, more reliable posterior, at a proportional number of extra model evaluations. |
| Posterior samples | SIR | How many of the prior draws to resample, with replacement, into the reported posterior. This is resampling only -- it adds no further model evaluations. |
| Chain length | MCMC | The total number of steps the chain takes. |
| Burn-in | MCMC | Early chain steps discarded before the chain has settled near the posterior. |
| Step size | MCMC | The proposal step, as a fraction of each calibrated input's Low/High range. Too small mixes slowly; too large rejects most proposals. Aim for roughly 15-50% acceptance, shown after running. |

The result panel reports a diagnostic alongside the posterior summary -- effective sample size for SIR (how many of the resampled posterior draws are effectively independent, out of the prior draws taken), acceptance rate for MCMC (the fraction of proposed steps the chain accepted) -- plus a plain converged/needs-review flag. A very low effective sample size or an acceptance rate far outside the 15-50% guideline is worth a re-run with adjusted settings before trusting the posterior.

#### Applying a result

Local Optimization, Random Search, and Grid Search show a before/after table of each calibrated input's initial and calibrated value; a Markov model additionally shows a chart overlaying the calibrated model's trace against the target points, while a decision tree shows a target-vs-predicted table instead (a tree has no cycle-by-cycle trace to chart). One button, "Apply calibrated values", writes the calibrated values into the model's real Parameters/Variables; "Discard result" clears the run without changing the model.

A Bayesian result adds a second way to apply it. "Apply posterior mean as base case" writes just the posterior mean into each calibrated input's base value, the same as the point-estimate algorithms' Apply button. "Apply posterior to PSA (recommended)" instead writes the whole posterior into that input's row in PSA Parameters, fitting a Beta distribution when the posterior mean falls strictly between 0 and 1 and a Normal distribution otherwise (by matching mean and standard deviation), upserting the row so re-applying after a later calibration run updates the same PSA entry rather than duplicating it. This is the recommended choice whenever the model already uses PSA, since it carries the calibration's own uncertainty forward into every future probabilistic run instead of discarding it at a single point estimate.

### PSA Setup

Probabilistic Sensitivity Analysis draws each configured input from a distribution on every iteration instead of holding it at its base value. PSA Parameters is the table of inputs varied this way: each row targets either a Parameter/Variable by name, or one shape/scale slot of a Distribution, and carries its own PSA distribution type (Beta, Normal, Gamma, ...) and values (e.g. mean, se).

#### Correlation

Two PSA Parameters rows can be linked with a pairwise correlation coefficient so they no longer vary independently across iterations. This is for a simple two-way relationship; for three or more parameters that must move together under a shared structure, use Correlated Distribution instead.

#### Correlated Distribution

A Correlated Distribution group draws several PSA Parameters rows jointly from a single named distribution, instead of independently. Two kinds exist in DECIDE: Dirichlet and Multivariate Normal. A decision tree has no Markov transitions, so only **Multivariate Normal** applies here -- Dirichlet groups a Markov state's outgoing transitions and is covered in Part C.

Pick two or more existing PSA Parameters rows, set each one's mean and standard deviation, and fill in the pairwise correlation between every pair of members. An inconsistent correlation matrix (one with no valid joint distribution) is automatically repaired to the nearest valid one before sampling.

Applying a group generates one set of jointly-drawn values per PSA iteration and stores them against the group; the group panel shows Applied / Not applied so it's clear when a change (a different member list, edited mean/SD/correlation) still needs to be applied before the next PSA run picks it up. A Correlated Distribution group only affects the PSA iterations that come after it is applied -- the model's own base-case values are unchanged.

### Value of Information (EVPI/EVPPI)

Probabilistic Sensitivity Analysis (PSA) draws each configured parameter from its distribution across a set number of iterations and records the resulting cost/effect for every strategy. Alongside the usual CE plane, CEAC/CEAF, and convergence diagnostics, the PSA results panel (shared across the decision tree and Markov model types) reports the Expected Value of Information -- the standard HTA/payer measure of how much uncertainty is costing the decision, and therefore whether further research would be worth funding.

#### EVPI

EVPI (Expected Value of Perfect Information) is the per-patient value of resolving all PSA uncertainty at once: the gap between the best outcome achievable if every parameter's true value were known in advance and the best outcome achievable under current uncertainty. The panel plots EVPI across a grid of willingness-to-pay thresholds centred on the model's own WTP setting, with a per-patient EVPI value reported at that threshold.

#### EVPPI by parameter

EVPPI (Expected Value of Partial Perfect Information) breaks EVPI down by PSA parameter, ranking which individual uncertainties are actually worth resolving through further research. Each parameter's EVPPI is estimated by regressing each strategy's net monetary benefit against that parameter's sampled values (the Strong/Oakley regression method) and measuring how much of the overall EVPI that regression alone would explain. The polynomial degree used for each parameter's regression is chosen automatically via k-fold cross-validation (falling back to a linear fit when a model has too few PSA iterations to cross-validate meaningfully) rather than fixed in advance, and is shown in the table's Degree column.

#### EVPPI for a group of parameters

Select two or more PSA parameters in the "EVPPI for a group of parameters" multiselect to estimate their joint EVPPI -- the value of a single study that resolved all of them together, which is not generally the sum of their individual EVPPI above, since parameters can interact in how they affect net monetary benefit. Keep a group small (2-4 parameters) relative to the number of PSA iterations, since the number of regression coefficients grows with both group size and polynomial degree; a group EVPPI estimated from too few iterations should be interpreted cautiously.

#### Population EVPI/EVPPI

EVPI and EVPPI are per-patient by default. The "Population (annual value of further research)" inputs -- annual eligible population, time horizon in years, and discount rate -- scale every EVPI/EVPPI value (including the grouped EVPPI) up to the total population the decision will affect, using the same discounting convention the model's own cost calculations use:

Effective population = sum over each year of the time horizon of (annual eligible population / (1 + discount rate)^year), with the first year undiscounted.

This is the number that is actually comparable to the cost of a proposed further study -- an HTA body decides a study is worth funding when its cost is below the population-scaled EVPI (or the population-scaled EVPPI of the specific parameters that study would resolve). Leaving the annual population at 0 disables population scaling and shows only the per-patient values.

## Part C -- Markov Model

### Engine variables

DECIDE provides two engine-owned variables in expressions. They are available
without defining a parameter or variable with the same name.

| Variable | Meaning |
| --- | --- |
| `_cycle` | The model cycle, starting at `0`. A decision tree is evaluated once with `_cycle = 0`. In a Markov model, `_cycle` increases by one for every simulated cycle. |
| `_tenure` | The number of consecutive cycles spent in the current Markov state, starting at `1`. Decision trees evaluate with `_tenure = 1`. |

#### Markov tenure states

Enable **Tenure applied** on a Markov state and set **No. of tenure** to model
time spent in that state. DECIDE represents the state internally as temporary
tunnel stages:

- Entry into the state starts at `_tenure = 1`.
- Each self-loop advances to the next tenure stage.
- The final stage retains the self-loop, so its `_tenure` remains at the
	configured maximum.
- All transitions to another state are available from every tenure stage. A
	transition to another state starts that destination state at `_tenure = 1`.

For a tenure-enabled state, costs apply as follows:

- **Initial cost** applies at `_tenure = 1` to everyone in the state in that
	cycle, including the portion that leaves during that same cycle. That
	portion receives only Initial cost (no Terminal cost).
- **Recurring cost** applies from `_tenure = 2` to the cohort portion that
	remains in the state for the next cycle.
- **Terminal cost** applies from `_tenure = 2` to the cohort portion that leaves
	the state in the next cycle. It replaces recurring cost for that portion
	(and is not added to Initial cost at `_tenure = 1`).
- To reproduce a plain per-cycle state cost `c` (as in TreeAge state rewards),
	enter `c` as Initial, Recurring and Terminal cost.
- At the end of the time horizon, a cohort that remains in the state receives
	recurring cost rather than terminal cost.

Cost and effect expressions inherit the original state context and can use
both `_cycle` and `_tenure`.

### Markov Editor

*Pending -- the canvas mechanics for Markov (State and Transition editing, Strategies, absorbing states, tenure settings) have not yet been verified against the current app UI for this version of the manual. This subsection will be filled in a follow-up pass. See Engine variables above for how tenure-enabled states are represented in the expression engine, and File panel / Decision Tree Editor in Part B for the mechanics shared with the tree editor (Save/Save as/Open, Autosave and Restore).*

### File panel

Same as Part B -> File panel.

### Model Calibration

Same mechanism as Part B -> Model Calibration. The one difference is the shape of a target row: for a Markov model each row is a (Cycle, State, Value, Weight) point along the model's cohort trace, and Applying a result additionally shows a chart overlaying the calibrated model's trace against the target points -- both already described in Part B's Target data and Applying a result subsections.

### PSA Setup

PSA Parameters and Correlation work exactly as in Part B -> PSA Setup. Correlated Distribution's **Dirichlet** kind is available here, since a Markov model's states have outgoing transitions to group:

| Kind | Use for | Draws |
| --- | --- | --- |
| Dirichlet | A Markov state's outgoing transition probabilities, so they vary independently but always sum to exactly 1 on every PSA iteration | One row per outgoing transition, each a bare Parameter reference (no formula, no complement marker) |
| Multivariate Normal | Any two or more PSA Parameters rows that should move together under a general correlation structure | Two or more existing PSA Parameters rows, picked by name (see Part B for detail) |

**Dirichlet.** Pick the Markov state; DECIDE finds its outgoing transitions automatically and requires every one of them to be a plain Parameter or Variable reference (a formula, or a state using a complement/"remaining probability" marker on one of its transitions, cannot be part of a Dirichlet group -- a complement transition is incompatible with Dirichlet sampling, since the group's own draws already sum to 1 and the complement would then always resolve to 0). Set a concentration value (alpha) for each outgoing transition; alpha's relative sizes set the group's average split, and its overall scale sets how tightly the draws cluster around that split. If any transition is not yet a PSA Parameters row, DECIDE offers to add it with a starting Beta distribution before the group can be applied.

Applying a group generates one set of jointly-drawn values per PSA iteration and stores them against the group; the group panel shows Applied / Not applied so it's clear when a change (a different state, a different member list, edited alpha) still needs to be applied before the next PSA run picks it up.

A Markov model's Dirichlet grouping applies to the currently open Strategy's transitions; it does not automatically extend to a state of the same name in a different Strategy.

Only Dirichlet and Multivariate Normal joint sampling are available for PSA today. A Multinomial-style joint distribution over discrete outcome counts is not yet implemented; it is a planned addition for a future model type where the outcomes it would apply to are counts of simulated events rather than continuous Markov transition probabilities.

### Strategy comparison and ICER

Base case results list the strategies from the cheapest to the most expensive. The cheapest is the reference. Each of the others is compared with the **previous strategy that is not dominated**, not simply with the row above it:

- **Dominated:** a strategy that costs more than a cheaper one but adds no effect. It is shown as *Dominated* and is never used as the comparator of the next strategy.
- **Extendedly dominated:** a strategy whose ICER against its comparator is higher than the ICER of the next, more effective strategy. It is shown as *Extendedly dominated*.

Incremental cost and incremental effect are measured against that comparator, and the ICER is their ratio. The same table is used for Decision Tree, Markov and Microsimulation models.

### Value of Information (EVPI/EVPPI)

Same as Part B -> Value of Information (EVPI/EVPPI). The PSA results panel and its EVPI/EVPPI reporting are shared unchanged between decision tree and Markov models.

## Part D -- Partitioned Survival Model (PSM)

A partitioned survival model (PSM) splits a cohort into three states -- **pre-progression** (progression-free), **post-progression** and **dead** -- directly from two survival curves, progression-free survival (PFS) and overall survival (OS). There are no transition probabilities: the proportion in each state at every time point is read off the curves, and costs and effects accumulate on those proportions. In DECIDE a PSM is a **PSM node** inside a decision-tree document, so the Decision node above it provides the strategies and everything Part B says about Parameters, Variables, Distributions, Tables, Sensitivity Analysis and PSA applies unchanged.

### How a PSM node is calculated

For each cycle `c = 0, 1, ..., N` (N = Cycles), at elapsed time `t = c / cycles per year` (years):

| State | Proportion of the cohort |
| --- | --- |
| Pre-progression | `S_PFS(t)` |
| Post-progression | `max(0, S_OS(t) - S_PFS(t))` |
| Dead | `1 - S_OS(t)` |

Each state has a per-cycle cost and effect (the branch's cost/effect increment plus the value on its Terminal node). The cycle's reward is the sum over the three states of proportion x rate, discounted by `1 / (1 + rate)^t`, where the rate is the annual rate from Model properties and `t` is in years. Cycles `0 ... N-1` are summed; with **half-cycle correction** switched on, the first and last time points count half (cycle 0 weight 0.5, cycle N weight 0.5). Cost and effect rates are re-evaluated in every cycle, so they may use `_cycle` (for example a drug dose that stops after cycle 35: `if(_cycle<35, cDrug, 0)`). `_cycle` starts at 0; `_tenure` is always 1 in a PSM.

A curve's time axis is in the **time unit** you choose for it (year, month, week or day), so parameters can be typed exactly as published; DECIDE converts elapsed years to that unit before evaluating the curve. **Cycles per year** is separate: it sets how long a cycle is.

### File panel

Same as Part B -> File panel. In addition, the PSM sidebar has an **Export code** panel (see "Export code" below).

### PSM Editor

The canvas is the Tree canvas with one extra node type. A typical model is Decision -> one PSM node per strategy -> three state branches.

- **Strategies:** branches leaving the Decision node. A strategy can also lead to a Chance node whose branches end in PSM nodes.
- **PSM node:** select it on the canvas and use **Edit node** (right of the canvas) to set the PFS and OS curves, **Cycles**, **Cycles per year** and **Half-cycle correction**.
- **State branches:** a PSM node has exactly three branches with fixed roles -- Pre-progression, Post-progression and Dead -- each ending in a Terminal node. The branch's cost increment and effect increment, plus the Terminal's cost and effect, are the per-cycle rates of that state (Dead is usually 0). Use expressions with `_cycle` for time-varying costs; utilities for a cycle that is shorter than a year are entered per cycle (for example `0.804*3/52` for 3 weeks).
- **Parameters:** curve parameters and rates may be numbers or expressions that reference Parameters and Variables (Part A), so they can be varied in Sensitivity Analysis and PSA.

#### Survival distributions

| Distribution | Parameters | S(t) |
| --- | --- | --- |
| Exponential | rate | `exp(-rate*t)` |
| Weibull | shape, scale | `exp(-(t/scale)^shape)` |
| Log-normal | meanlog, sdlog | `1 - Phi((ln t - meanlog)/sdlog)` |
| Log-logistic | shape, scale | `1 / (1 + (t/scale)^shape)` |
| Gompertz | shape, rate | `exp(-(rate/shape)*(exp(shape*t) - 1))` |
| Generalized gamma | mu, sigma, Q (Prentice parametrisation) | regularised incomplete gamma; Q = 0 is the log-normal |

`t` is in the curve's time unit. Parameters follow the flexsurv conventions described in Part A (Distribution types and parameters), so values fitted with flexsurv can be typed directly; values published in another convention (for example a Weibull given by a rate or a proportional-hazards form) must be converted with the table there first.

#### Fit from KM / patient data

You do not need the NMA module to get a curve. Open **Fit from KM / patient data** inside a curve's editor:

1. Choose the data type: **patient-level data** (rows of time and event: 1 = event, 0 = censored; paste or upload a CSV) or **digitised Kaplan-Meier points plus the numbers-at-risk table** (DECIDE reconstructs individual times with the Guyot algorithm).
2. Choose the time unit of the data and click **Fit all distributions**.
3. DECIDE fits all six distributions by maximum likelihood (right-censoring is handled) and lists log-likelihood, AIC, BIC and the largest gap between the fitted curve and the Kaplan-Meier curve, with a chart overlaying every fit on the Kaplan-Meier curve.
4. Pick a distribution and click **Use this fit**. The curve's distribution, parameters and time unit are filled in, and the fit (summary, number of patients and events, candidates, thinned KM points) is stored with the curve; it appears as the curve's **Source** in the Technical Report. Editing the curve by hand afterwards removes the stored fit.

Choosing between the candidates is a judgement: inspect the overlay and the long-term extrapolation, not AIC alone. The three panels below (Curve library, Extrapolation & hazard, Plausibility checks) are there for that. Piecewise or spliced extrapolation and hazard ratios/covariates are not available yet.

#### Curve tools: library, extrapolation and plausibility

Three panels work on the PFS and OS curves of a PSM node. In the desktop app they are in the **Setting** menu; in the browser they are sections of the sidebar. Each starts with a **Curve** selector (for example `psm_1 - OS`) that picks which curve of which PSM node you are working on. The same candidates feed all three panels.

**Curve library.** The library is a shortlist of candidate curves kept with one PFS or OS curve, so that you can compare several fits and choose the one used in the model, and keep a record of why.

1. The line *Curve in use* shows the distribution the model is currently using.
2. To fill the library, click **Add the N candidates from the last KM fit** (shown only after a *Fit from KM / patient data* run; it adds every fitted distribution with its AIC and BIC), or give the current curve a **Name** (for example "Weibull (best AIC)"), optionally write a **Rationale for choosing this curve**, and click **Save current curve as candidate**. A curve whose parameters are named model parameters rather than numbers cannot be saved as a candidate.
3. The table lists each candidate's name, distribution, parameters, time unit, AIC, note and whether it is **In use**. Select a candidate and click **Use this candidate** to copy its distribution, parameters and time unit into the curve (the model then runs with it), or **Remove from library**.

Saving a candidate under an existing name replaces it. The library is saved with the model file.

**Extrapolation & hazard.** Compares the current curve, the library candidates and the candidates of the last KM fit side by side over the model horizon. Enter the **Model horizon** and the **Observed follow-up** (taken from the stored Kaplan-Meier curve when there is one). The table gives, for every curve, survival at fixed landmark years, the median, the mean survival up to the horizon, the *share beyond data* (the part of that mean that lies after the observed follow-up, so it is extrapolation rather than data) and survival at the horizon. Two charts overlay survival (with the Kaplan-Meier curve in black and the end of follow-up dashed) and hazard per year.

**Plausibility checks & background mortality.** Lists warnings for each candidate: a fit that did not converge, a large share of the mean survival lying beyond the observed follow-up, many patients still alive at the horizon (so the mean is truncated), a Gompertz curve with a plateau, and (when a life table is entered) survival above the general population or a hazard below background mortality. For an OS curve it also warns if the PFS curve exceeds it. You can paste a **general-population life table** (one line per row: age, annual probability of death between 0 and 1) and the **Age at start**. DECIDE then compares each candidate with the general population and shows the mean survival with a background-mortality floor applied after a chosen year (from that year the hazard is the larger of the fitted hazard and general-population mortality). The floored curves are for comparison only; the model itself still uses the curve as typed.

### Model properties

Outcomes (cost, effect and any extra rows) and the Discount table work as in Part B; the discount rates are annual and are applied to elapsed time in years, whatever the cycle length.

### Validation

Before Run, DECIDE checks that every PSM node has its three state branches ending in Terminal nodes, that each curve has a known distribution and positive parameters, and whether PFS exceeds OS (a progression-free cohort cannot be larger than the cohort still alive). A missing branch, unknown distribution or non-positive parameter blocks the run until it is fixed. Curves that cross by a very small amount, which happens when PFS and OS are fitted independently, are reported as a review item (warning) in the self-checks and do not block the run.

### Base case results

Base case results show, per strategy, total cost, total effect, incremental values and the ICER (CEA) exactly as in Part B. For paths ending in a PSM node the Paths table shows that node's discounted totals.

### Survival Curves / State / Time Report

The **Report** button opens a panel for one PSM node:

- **Survival Curves:** PFS and OS by cycle.
- **State Report:** the node's totals split by state -- area under the occupancy curve (in cycles) and cost and effect for each state, undiscounted and discounted. Each cycle's own rates and the half-cycle weights are used, so the **Total** row equals the Run/ICER figure for the node.
- **Time Report:** one row per cycle with the three occupancies and the cycle's cost and effect.

### Result Dashboard

The dashboard is the one described in Part B (Base case results, Tornado, PSA cards) with two extra PSM cards:

- **PSM - PFS / OS curves by strategy:** choose **All strategies** (one chart; colour identifies the strategy, solid lines are PFS, dashed lines are OS) or **Single strategy** (pick one strategy; blue is PFS, red is OS). **Curves** selects PFS and OS, PFS only or OS only. The choices are saved with the card.
- **PSM - State occupancy (stacked):** one panel per strategy showing the proportion pre-progression, post-progression and dead over time.

A new PSM model starts with Base case results and these two charts on its dashboard. When a strategy reaches several PSM nodes (for example under a Chance node) each node is drawn separately, labelled with the strategy and the node id. The dashboard exports to PDF.

### Documentation exports

Under Base case results, **Documentation exports** produce material for review and for a technical appendix. They are built from the same calculation as Run, and DECIDE refuses to build them if its own replay of the model does not reproduce the engine's totals.

- **Model transparency Excel:** a workbook that recomputes the model with live formulas. Each PSM node has its own sheet ("PSM <node id>") with the curve parameters, survival, occupancy, state rates, discounting and the half-cycle correction by cycle, so changing a cell recalculates the node. The Tree Paths sheet links each path that ends in a PSM node to that sheet. Rate expressions that cannot be translated to an Excel formula are shown as amber cells with their resolved values and a comment. All six distributions are supported.
- **Technical Documentation Report (Word + Excel):** model structure, parameter table (parameters with no Source are highlighted), survival curves with their sources and curve-fitting evidence (AIC/BIC, fit to the Kaplan-Meier data), figures (survival curves with the KM data, overlay of strategies, stacked occupancy), occupancy and time-in-state tables, base case results and ICER, and the validation self-checks. The report is not built while a self-check of error severity fails.

#### Validation self-checks

Each PSM node is checked on every report, and the results are listed in the report:

| Check | Meaning |
| --- | --- |
| PFS curve does not exceed OS curve | Review item (warning) if the curves cross |
| S(0) = 1, curves within [0, 1] and non-increasing | Each curve is a valid survival function |
| Occupancies sum to 1 | Review item if curves cross |
| Replay equals the engine | The cycle-by-cycle replay reproduces the node's cost and effect |
| Independent recomputation | A separate implementation of the survival formulas and the sum reproduces the totals |
| Zero reward gives zero | With all rates 0 the totals are 0 |
| Dead-only reward | A reward only in Dead equals the sum of the (1 - OS) discounted |
| Reward 1 in every state | The total equals the sum of the discount factors |

Failures of the error-severity checks stop the report; "Review" items are model-quality notes and do not.

### Export code

The **Export code** panel in the PSM sidebar writes the model as a stand-alone script in **Python, R, Stata (Mata) or JavaScript**. Each PSM node becomes one function that loops over the cycles exactly as described above and returns its discounted cost and effect, and the script prints each strategy's total cost and effect (and the ICER for two strategies). Survival functions are included in the script. Python and JavaScript output has been checked against DECIDE's results; R and Stata output follows the same logic but should be run once on your machine to confirm.

Limits: a generalised gamma curve must use the Prentice (mu, sigma, Q) form; an expression that depends on `_cycle` and cannot be written in plain arithmetic / `if()` stops the export with a message instead of being fixed at one value.

### Model Calibration, Sensitivity Analysis, PSA and Value of Information

These work as in Parts B and C. Curve parameters, state costs and utilities, and `Cycles` inputs expressed through Parameters can be varied in Sensitivity Analysis and given distributions in PSA; correlated sampling and EVPI/EVPPI are shared.

### Worked example

`samples/psm_cai2021_sample.json` reproduces the structure of a published pembrolizumab + chemotherapy versus chemotherapy model (Cai et al., Front Oncol 2021): two strategies, each with a PSM node (log-logistic and generalised gamma PFS, log-logistic OS, curves in months), 174 cycles of 1/17.39 year, half-cycle correction and 5% annual discounting, with drug costs that stop after fixed numbers of cycles using `_cycle`. Open it, press Run, then open Report, Dashboard and the documentation exports to see every feature above. The base case is close to, but not an exact match for, the paper's Table 3.

## Part E -- Microsimulation

A microsimulation follows **individual patients**, one at a time, instead of a cohort average. Each simulated patient starts in a health state, and at every cycle a random draw decides the next state from that patient's current transition probabilities. Costs and effects accumulate per patient, and the results DECIDE reports are the **means across patients**. Because each patient is simulated on their own, a microsimulation can carry history that a cohort model cannot (time in a state, prior events, patient-specific values), at the price of simulation noise and a longer run time.

A Microsim model is built on the same canvas as a Markov model (Part C): a Decision node fans out into **strategies**, each strategy has its own root **M** node, and health states and transitions hang below it. Parameters, Variables, Distributions and Tables use the expression language of Part A. What is specific to Microsim is described here.

### How a Microsim is calculated

For each of the **N patients**, and for each cycle `c = 0, 1, ..., Cycles - 1`:

1. The patient's current health state, the engine variables (`_cycle`, `_tenure`) and the current value of every **Tracker** (see Trackers below) are set up.
2. The outgoing transition probabilities of the current state are resolved (a `#` complement takes whatever probability is left).
3. The state's **cost** and **effect** for the cycle are computed and discounted: `value / (1 + rate)^c` (cycle 0 is not discounted). With **half-cycle correction** switched on, half of the difference between the last and the first cycle's reward is added, as in the Markov cohort engine.
4. The next state is drawn at random from the resolved probabilities.
5. Tracker update formulas run, and the reset rules of a state the patient has just entered are applied.

The patient's totals are averaged over all patients to give the strategy's expected cost and effect.

- **Initial state.** Each patient's starting state is drawn from the **Initial probability** of the health states (they must sum to 1; one state may use `#`).
- **Random numbers.** Every patient has an independent random-number stream derived from the **Random seed**. The same patient number uses the same stream in every strategy (common random numbers), so the difference between two strategies is not inflated by sampling noise. The same model, N and seed always return exactly the same results.
- **Precision.** Results carry a Monte Carlo standard error (MCSE): the standard deviation of the per-patient values divided by the square root of N. Quadrupling N halves it. A microsimulation converges to the equivalent cohort model as N grows.
- **Tenure states.** A health state with **Tenure applied** works as in Part C (Initial, Recurring and Terminal cost). Because a single patient cannot know in advance whether it is about to leave the state, the cost blends the three components using the probability that the patient leaves in this cycle, exactly as the cohort engine does.

### Health states and branch nodes

Only the **direct children of the strategy's M node** are health states. A patient occupies one of them for a whole cycle, and each has a **per-cycle Cost and Effect**.

Nodes drawn below a health state (chance branches and terminal nodes) are **not** health states. They are decision points inside a single cycle:

- A patient who reaches a branch node is resolved through it to a health state **within the same cycle**; no time passes.
- A **terminal node ends the cycle**: the patient is in the health state that the terminal stands for, and in the next cycle moves on from that health state. A terminal is therefore not absorbing. Use **Relabel as terminal** on the canvas to point a terminal at the health state it represents (a terminal with the same name as a health state is matched automatically). Only the real **Dead** state is marked as absorbing.
- Branch and terminal nodes have **no per-cycle cost or effect**. Their Edit state box shows only **Incre_$** and **Incre_E** (for example QALY): a one-time cost or effect, charged once, in the cycle in which the patient passes along the transition into that node. These are the same numbers as the Incre values on the canvas edges.

For example, in a Sick-Sicker model Sick has three branches (recover, progress, die) plus a "stay sick" terminal. A patient who "stays sick" ends the cycle in Sick, and next cycle starts from Sick again.

### File panel

Same as Part B -> File panel. The model's N patients, Random seed and Store N sample patients setting are saved with the file. Autosave and the Restore banner work as for the other editors.

### Microsim Editor

- **Toolbar:** auto layout, **Run**, **SA**, **PSA**, **Dashboard** and Model Calibration. Each opens its panel below the canvas.
- **Strategies:** as in Part C. Each strategy has its own health states and transitions; Parameters, Variables, cycles, discounting and N are shared by all strategies.
- **Canvas view:** **State Diagram** (the editable canvas) or **Patient Journey** (see below).
- **Edit state:** select a node, edit it on the right and press **Apply**. A health state has Name, **Cost** (or Initial / Recurring / Terminal cost when **Tenure applied** is on, with **No. of tenure**), **Effect**, **Initial probability** and **Absorbing**. A branch or terminal node has Name, Incre_$ and Incre_E (see above).

#### Model properties

| Setting | Meaning |
| --- | --- |
| Number of cycles | Number of cycles simulated. The reported trace has one more row than the number of cycles because it starts at cycle 0. To reproduce a source that reports time points 0 to 30, use 31 cycles. |
| Cycles per year | Length of a cycle. Used for the discount unit and by rate/probability conversions. |
| Half-cycle correction | Applies the half-cycle correction described above. Leave it off to reproduce a model that does not use one. |
| N patients | Number of simulated patients. Default 1,000. Larger N gives a smaller MCSE and a longer run. |
| Random seed | Seed of the random-number streams. Changing it gives a different, equally valid sample. |
| Store N sample patients | How many patients (the first N by index) have their full cycle-by-cycle path recorded for the Patient Trace view. Can be as large as N patients. 0 turns it off. |
| Outcomes, Discount | As in Part B. Discount rates can be entered per cycle or per year. |
| Analysis | Analysis type, effect objective, baseline strategy and willingness-to-pay threshold, as in Part C. |

### Trackers

A **Tracker** is a number carried by each patient through the simulation, for example "number of events so far" or "time on treatment". It can be used in any probability, cost or effect expression of that patient.

| Column | Meaning |
| --- | --- |
| Name | The tracker's name, used in expressions. Must be unique. |
| Initial value | The value at cycle 0. |
| Update formula (per cycle) | Evaluated once per cycle after the patient's transition; the result becomes the new value. For example `n_events + 1`. The formula sees the values from the start of the cycle. |
| Reset rules | `state=value, ...`: when a patient enters the named state, the tracker is set to that value. |
| In output | Include the tracker in the Tracker summary of the results. |

**Tracker reset timing** chooses when a reset takes effect: **On entering the new state**, or **One cycle after entering**. Use **Add tracker** to add a row, the bin icon on a row to delete that row, and **Update** to confirm the table.

### Resampling distributions

In the Distribution table, **Resample** says how often a random value is drawn: **Per model** (once for the run), **Per PSA iteration**, **Per patient** (each patient gets their own draw, for example patient-specific age or risk) or **Per cycle**. Per-patient and per-cycle draws are what let a microsimulation represent patient heterogeneity.

### Validation

Before Run, DECIDE checks the model and reports the first problem found, which blocks the run until it is fixed. The most common messages are:

| Message | What to check |
| --- | --- |
| `initial_distribution_sum` | The Initial probabilities of the health states must add up to 1 (at most one may use `#`). |
| `transition_sum` / `missing_transitions` | The outgoing probabilities of a state add up to more than 1, or a health state has no transitions and is not absorbing. |
| `unresolved_alias` | A terminal node does not lead anywhere. Relabel it as a terminal of a health state, or give it the same name as the health state it represents. |
| `non_real_state_loop` | Branch nodes lead back to each other without reaching a health state. |
| `no_real_states` | The strategy has no health state directly below its M node. |
| `invalid_n_patients` | N patients must be at least 1. |
| `duplicate_tracker`, `duplicate_state` | Names must be unique. |

### Run and Base case results

Press **Run** to simulate every strategy with the full N and open **Base case results**:

- **Strategy comparison:** total cost, total effect, incremental cost and effect, and ICER against the baseline, exactly as in Part C.
- **Trace:** a line chart of the proportion of patients in each health state at each cycle, and **one table** with the same rows: the proportions, and the raw and discounted cost and effect of each cycle. The table can be downloaded as CSV or Excel. With several strategies, **Strategy to trace** chooses which one is shown.
- **Tracker summary:** mean, 10th, 50th and 90th percentile of each tracker's final value, and the distribution of per-patient total cost and effect.

How long a run takes depends mainly on N, the number of cycles and the number of strategies. On the Sick-Sicker example one strategy takes about 0.6 seconds at N = 1,000, 6 seconds at N = 10,000 and 18 seconds at N = 30,000; N = 100,000 takes about a minute per strategy. Use a small N while building the model and the full N for the final run.

### Patient Journey

Choose **Patient Journey** under Canvas view to look at the simulated patients instead of the model diagram. A **Strategy** selector above the view chooses which strategy to show.

- **Population Flow:** how the population moves between health states from cycle to cycle, as straight flow bands. The view is drawn from a simulation of at most 5,000 patients so that it stays fast; a note under the chart says so. **Run** always uses the full N.
- **Patient Trace:** one line per patient showing the state they are in at every cycle. The path of up to **Store N sample patients** patients is recorded. The browser draws up to 100 evenly spaced patients; **Prepare CSV** then **Download patient paths (CSV)** gives the path of every stored patient.

The legend lists only the health states.

### Documentation exports

Press **Build Technical Documentation Report** under Base case results to produce a Word report and a companion Excel workbook. It is built from the results already on screen; DECIDE also runs the checks below on the model. The report contains: model structure and, when you have clicked the camera button on the canvas, the diagram; simulation settings (N, seed, cycles, discounting); the parameter table (parameters with no Source are highlighted); expected cost and effect with their MCSE and the ICER; a comparison with the equivalent cohort model; the validation self-checks; and a few cycles of each strategy's trace (the full trace is in the Excel file, one sheet per strategy). Unlike the Markov and Tree workbooks, the Excel file holds values, not live formulas, because a microsimulation cannot be recalculated in a spreadsheet.

#### Validation self-checks

| Check | Meaning |
| --- | --- |
| Patients are conserved | The proportions across states add up to 1 in every cycle. |
| No negative state occupancy | No state ever has a negative share. |
| Absorbing states never lose patients | Once patients enter an absorbing state they stay. |
| Mean of per-patient totals equals the total | The reported cost and effect are the mean of the per-patient values. |
| Per-cycle values add up to the totals | The discounted per-cycle cost and effect sum to the totals (when half-cycle correction is off). |
| Fixed seed reproduces results | Two runs with the same seed give identical results. |
| Agreement with the cohort model | The microsimulation is within 3 Monte Carlo standard errors of the same model run as a cohort (Review item if it is not). |

A failing check of error severity stops the report; "Review" items are model-quality notes and do not.

### Sensitivity Analysis, PSA, Dashboard and Model Calibration

These panels work as in Parts B and C. Every scenario, iteration or calibration step **runs the whole simulation again**, so they take far longer than one Run: use a small N (for example 1,000) and few iterations while setting them up, and keep Cycles and the number of strategies in mind. A Distribution with Resample set to **Per PSA iteration** is drawn once per PSA iteration and shared by all strategies in that iteration.

### Worked example

`samples/microsim_darth_sicksicker_sample.json` is the model a new Microsim editor opens with. It is the Sick-Sicker model of Krijkamp et al., Medical Decision Making 2018;38(3):400-422 (https://doi.org/10.1177/0272989X18754513): four health states (Healthy, Sick, Sicker, Dead), two strategies (No treatment, Treatment), a Parameters table (costs, utilities and probabilities entered by name), **31 cycles**, **no half-cycle correction**, 3% discounting of costs and effects, and N = 100,000. The paper's Table 2 reports the deterministic cohort model and the microsimulation; DECIDE gives:

| Result | Publication (cohort model) | DECIDE (cohort model) |
| --- | --- | --- |
| No treatment | $75,976 and 15.83 QALY | $75,976 and 15.839 QALY |
| Treatment | $141,623 and 16.40 QALY | $141,623 and 16.400 QALY |
| ICER | $116,901 per QALY | $116,901 per QALY |

The microsimulation of the same model agrees with these within Monte Carlo error: at N = 10,000 the paper itself reports $75,790 and $141,211, and at N = 100,000 $75,996 and $141,644. Open the example, press **Run**, then open **Patient Journey**, the **Technical Documentation Report** and **PSA**.
