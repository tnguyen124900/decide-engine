# DECIDE expression language

The formula language used for every Cost, Effect, Probability, Parameter and Variable definition in the engine.

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
- **Renaming is automatic.** Rename a Parameter, Variable, Distribution, Table or Tracker in its input table and every place that uses the old name is rewritten to the new one -- canvas formulas (branches, states, transitions), other Variables' definitions, Distribution parameters, trackers and rates, PSA setup and Sensitivity/Calibration selections -- in Decision Tree, Markov, PSM and Microsimulation alike. Text inside quotes (a table column such as `tbl["age"]`) and longer names that merely contain the old one are never touched. If the new name is already taken, nothing is rewritten and the duplicate-name warning shows.

Use a parameter for a direct numeric input, and a variable for a derived value:

```text
rate_AE_3yr = 0.5
prob_AE = RateToProb(rate_AE_3yr, 1, 1/4)
```

Then use `prob_AE` in a decision-tree branch probability, terminal cost/effect, Markov initial probability, state cost/effect, or transition probability.
