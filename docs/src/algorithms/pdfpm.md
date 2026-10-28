# PDFPM

`PDFPM` implements a partially derivative-free proximal method for composite
multiobjective optimization [Amaral2025PartiallyDerivativeFree](@cite).

```julia
result = PDFPM(evalf, A, delta, x0, PDFPM_options();
               evalJf=evalJf, lb=lower, ub=upper)
```

The method constructs quadratic component models, solves their proximal
subproblem with Ipopt, and uses an inner loop to test decrease and adjust
regularization. It updates component curvature matrices with a safeguarded
BFGS-like formula.

## Parameters and stopping

| Option | Default | Role |
| --- | --- | --- |
| `sigma` | `1.0` | Initial regularization; doubled after a rejected candidate |
| `epsilon` | `1e-10` | Proximal-step and sufficient-decrease parameter |
| `alpha` | `0.1` | Sufficient-decrease coefficient |
| `max_subproblem_iter` | `200` | Maximum candidate attempts per outer iteration |
| `lambda_factor` | `0.1` | Factor for finite-difference step selection |
| `lambda_min` | `1e-12` | Minimum finite-difference step |
| `step_tolerance_rule` | `:epsilon` | Thresholds of the proximal-step and decrease tests |

With the default `step_tolerance_rule=:epsilon`, a candidate passes the
decrease test when all objectives satisfy

```math
F_j(p)\le F_j(x)-\frac{\alpha\varepsilon^2}{2\sigma},
```

and the proximal criterion is ``\sigma\lVert p-x\rVert_2\le\varepsilon``.
With `step_tolerance_rule=:trace_scaled` and
``T=\max_j\operatorname{tr}(B_j)`` for the curvature matrices of the current
subproblem, the decrease test is

```math
F_j(p)\le F_j(x)-\frac{\alpha\varepsilon^2}{32\sigma(1+T)^2},
```

and the proximal criterion is ``\sigma\lVert p-x\rVert_2<\varepsilon/(4(1+T))``.
A step norm equal to the threshold stops under `:epsilon` and proceeds to the
decrease test under `:trace_scaled`.

The proximal criterion returns `:proximal_step_tolerance`. That return has `abs_theta=NaN`, since the model
measure was not recomputed at the returned candidate.

The default `stop_criteria=:pdfpm` also stops on `abs_theta <= opt_tol`.
`:NormProgress` combines that test with `norm_progress <= ftol` after an
iteration. The initial point is checked using `abs_theta` alone.
`max_subproblem_iter` limits this method's inner attempts, not Ipopt iterations.

## Traces of σ and T

With any `store_level` other than `:none`, entry ``k`` of `sigma_trace` and `max_tr_B_trace` belongs
to entry ``k`` of the other traces (`x_trace` only from `:medium` on), under both step tolerance rules:

| Entry | `sigma_trace` | `max_tr_B_trace` |
| --- | --- | --- |
| Initial point | `NaN` | `NaN` |
| Accepted step | σ of the decrease test that accepted it | ``T`` of the subproblem that produced it |
| Proximal-step exit | σ of the stopping test | ``T`` of the subproblem that produced the candidate |

Here ``T=\max_j\operatorname{tr}(B_j)`` is taken before the BFGS update that follows
an accepted step. No test precedes the initial point, so both values are `NaN`
there. A run that stops on a rejected candidate or a failure records no new entry.
Under `:epsilon`, ``T`` does not enter the tests but is recorded for comparison.

## Derivative inputs

Providing `evalJf` uses the analytical evaluator. Omitting it uses central
finite differences with

```math
\lambda=\max\left(\texttt{lambda\_factor}\,
\frac{\varepsilon}{\sigma\sqrt n},\;\texttt{lambda\_min}\right).
```

A numeric `lambda_factor` is clamped to `[0, 1]`. A function is called as
`lambda_factor(; iter)` and its return value is converted to the solver type.

!!! warning "Finite-difference precision"
    In Float32, the default perturbations can round back to the current
    coordinates and produce an incorrect zero Jacobian and stopping decision.
    For example, `2f0 + 1f-11 == 2f0`. Provide an analytical Jacobian for Float32
    solves. Perturbation accuracy also depends on coordinate scale and the
    objective evaluation; finiteness alone does not establish accuracy.

See [Results and stopping](@ref) for counters, limits, and failure behavior.
