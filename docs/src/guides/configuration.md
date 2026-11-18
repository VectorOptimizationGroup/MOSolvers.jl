# Configuration

Each method has an options constructor: [`CondG_options`](@ref),
[`ProxGrad_options`](@ref), or [`PDFPM_options`](@ref).

| Common option | Default | Meaning |
| --- | --- | --- |
| `max_iter` | `1000` | Maximum outer iterations |
| `max_time` | `3600.0` | Execution-time limit in seconds |
| `verbose` | `1` | `0` suppresses solver progress and termination summaries |
| `print_interval` | `1` | Number of iterations between progress rows |
| `store_level` | `:none` | Traces kept in the result: `:none`, `:low`, `:medium`, or `:high` |
| `max_cond_A` | `1e8` | Largest accepted ``\kappa_2(\tilde A_j)`` when `delta > 0` |
| `subproblem_tol` | `sqrt(eps(T))` | Tolerance of the subproblem solver (Ipopt; in CondG, HiGHS, which is limited to at least `1e-10`), for the element type `T` of `x0` |

CondG and ProxGrad add `opt_tol` (default `1e-4`), the strict tolerance on the
method's model measure; ProxGrad also adds `mu`. PDFPM adds `sigma`, `epsilon`, `alpha`,
`max_subproblem_iter`, `lambda_factor`, and `lambda_min`; see [PDFPM](@ref).
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

Callbacks run after the convergence check of an outer iteration, with `result.x`
and `result.Fval` at the accepted iterate. In PDFPM this comes before any doubling
of σ of the next iteration. They are not called at the initial point or on an
iteration that already satisfies a stopping criterion: PDFPM's Step 3 exit, or
CondG/ProxGrad's `abs_theta < opt_tol`, so with `opt_tol=0` these two visit every
accepted iterate. Read the initial and returned points from the result. During execution, `result.status` is `:running`; `total_time` is
filled on return. A callback should treat solver arrays as read-only and copy
any values it needs to retain.

## Traces

`store_level` selects the traces filled in the result. Stored arrays are copied,
and the level does not change iterates, evaluations, or termination.

| Level | Traces filled |
| --- | --- |
| `:none` | None |
| `:low` | `Fval_trace`, `abs_theta_trace`, `norm_progress_trace`; PDFPM also `sigma_trace` and `max_tr_B_trace` |
| `:medium` | `:low` plus `x_trace` |
| `:high` | Same as `:medium`; reserved for heavier traces |

`:low` costs one objective vector of length ``m`` and a few scalars per recorded
point; `:medium` adds one point of length ``n``.
PDFPM's `sigma_trace` and `max_tr_B_trace` are aligned entry by entry
with its other traces; CondG and ProxGrad leave them empty. See [PDFPM](@ref).
The initial progress is `NaN`, since no previous iterate exists. ProxGrad records
its initial `x` and `Fval` twice, while the initial measures are recorded once;
these trace arrays should not be assumed to share an index at initialization.

!!! compat "Breaking change"
    `store_level` replaces the former Boolean option `store_trace`, which was
    removed without deprecation. Use `store_level=:medium` for `store_trace=true`
    and `store_level=:none` for `store_trace=false`.
