# Configuration

Each method has an options constructor: [`CondG_options`](@ref),
[`ProxGrad_options`](@ref), or [`PDFPM_options`](@ref).

| Common option | Default | Meaning |
| --- | --- | --- |
| `max_iter` | `1000` | Maximum outer iterations |
| `opt_tol` | `1e-4` | Tolerance on the method's model measure |
| `stop_criteria` | Method name as a symbol | Model measure, or `:NormProgress` for measure and progress together |
| `ftol` | `1e-4` | Progress tolerance used by `:NormProgress` |
| `max_time` | `3600.0` | Execution-time limit in seconds |
| `verbose` | `1` | `0` suppresses solver progress and termination summaries |
| `print_interval` | `1` | Number of iterations between progress rows |
| `store_trace` | `false` | Store copies of the reported iterates and measures |

ProxGrad adds `mu`. PDFPM adds `sigma`, `epsilon`, `alpha`,
`max_subproblem_iter`, `lambda_factor`, `lambda_min`, and `step_tolerance_rule`;
see [PDFPM](@ref).
CondG and ProxGrad accept a separate `linesearch` keyword; see
[Line search](@ref backtracking).

## Numeric types and inputs

The element type of `x0` determines the solver's floating-point type. Options,
matrices, and vector bounds are converted at the solver boundary. Time values
use `Float64`. The solver copies `x0`.

Evaluation functions receive the solver's point vector and return objective
values or a Jacobian with the dimensions in [Problem formulation](@ref).
Float32 inputs do not make the underlying HiGHS and Ipopt computations Float32.
For finite-difference restrictions, see [PDFPM](@ref).

## Callbacks

Pass `callback(iter, result)` as a solver keyword. Returning exactly `true`
requests interruption; other return values continue the iteration.

Callbacks run after the convergence check of an outer iteration. They are not
called at the initial point or on an iteration that already satisfies a stopping
criterion. During execution, `result.status` is `:running`; `total_time` is
filled on return. A callback should treat solver arrays as read-only and copy
any values it needs to retain.

## Traces

With `store_trace=true`, the result contains `x_trace`, `Fval_trace`,
`abs_theta_trace`, and `norm_progress_trace`. Stored arrays are copied.
The initial progress is `NaN`, since no previous iterate exists. ProxGrad records
its initial `x` and `Fval` twice, while the initial measures are recorded once;
these trace arrays should not be assumed to share an index at initialization.
