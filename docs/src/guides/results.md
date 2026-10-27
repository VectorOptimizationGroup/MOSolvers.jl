# Results and stopping

The returned `MOSolvers.OptimResult` contains:

| Fields | Meaning |
| --- | --- |
| `x`, `Fval` | Returned point and its composite objective vector |
| `x_init`, `F_init` | Initial point and its objective vector |
| `iter` | Last recorded outer iteration |
| `abs_theta` | Absolute model measure at the recorded point; `NaN` if unavailable |
| `norm_progress` | Normalized infinity-norm change of the recorded point |
| `n_f_evals`, `n_Jf_evals` | Objective and Jacobian evaluator calls |
| `total_time`, `subproblem_time` | Total time and time spent in direction subproblems, in seconds |
| `status`, `message` | Stopping cause and explanatory text |

For consecutive recorded points, progress is

```math
\frac{\lVert x^k-x^{k-1}\rVert_\infty}
     {\max(1,\lVert x^{k-1}\rVert_\infty)}.
```

`n_f_evals` includes the calls used by finite differences; `n_Jf_evals` counts
requests for a Jacobian. `subproblem_time` counts direction subproblem solves,
not the support-function solves used to evaluate `h`.

## Termination causes

[`converged(result)`](@ref converged) reports whether one of the method's
stopping criteria was met. It does not certify a global optimum or make the
model measures of the three algorithms interchangeable.

| Status | Meaning | `converged(result)` |
| --- | --- | --- |
| `:optimality_tolerance` | Model measure satisfies `opt_tol`; includes initialization and CondG/ProxGrad's explicit zero-measure exit | `true` |
| `:optimality_and_progress_tolerance` | Both model measure and progress satisfy their tolerances | `true` |
| `:proximal_step_tolerance` | PDFPM satisfies `sigma * norm(p-x) <= epsilon` | `true` |
| `:max_iter` | Outer iteration limit reached | `false` |
| `:max_time` | Outer time limit reached | `false` |
| `:callback` | Callback requested interruption | `false` |
| `:max_subproblem_iter` | PDFPM exhausted its inner attempts | `false` |
| `:evaluation_failure` | An evaluator reported failure, including `h` during a search | `false` |
| `:subproblem_failure` | Direction subproblem failed | `false` |
| `:line_search_failure` | Search accepted no step | `false` |
| `:numerical_failure` | Required model data or outputs were nonfinite | `false` |

Use `status` for programmatic decisions and `message` for details. A backend
failure or backend time limit appears under `:subproblem_failure` or
`:evaluation_failure`, according to the operation. It does not establish
infeasibility of the original problem or imply that the outer time limit expired.

For the PDFPM proximal-step exit, `abs_theta` is `NaN`: the model measure was
not recomputed at the returned point. Initially uncomputed objective values
are also `NaN`.

## Limits and failures

The initial convergence test runs even with `max_iter=0` or `max_time=0`.
A convergence test completed on the last iteration precedes the callback and
loop exhaustion. A callback stop or specific failure retains its own cause.
Time checks occur between operations; they do not interrupt an active evaluator
or backend call.

The result retains the last fully recorded point if a trial or an operation
before the next result update fails. A point may have passed the decrease test
before a later Jacobian or subproblem failure prevents its result update.

Initial evaluator, callback, and direction-subproblem exceptions propagate.
CondG and ProxGrad catch exceptions from their composite evaluator during the
line search and report `:evaluation_failure`. Nonfinite trial objective values
are rejected by the search and may be followed by a successful smaller step.
