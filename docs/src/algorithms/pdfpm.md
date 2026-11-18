# PDFPM

`PDFPM` implements a partially derivative-free proximal method for composite
multiobjective optimization [Amaral2025PartiallyDerivativeFree](@cite).

```julia
result = PDFPM(evalf, A, delta, x0, PDFPM_options();
               evalJf=evalJf, lb=lower, ub=upper)
```

The method constructs quadratic component models, solves their proximal
subproblem with Ipopt, and uses an inner loop to test decrease and adjust
regularization. By default it updates component curvature matrices with a
safeguarded BFGS-like formula; `curvature=:zero` keeps them zero.

## Parameters and stopping

| Option | Default | Role |
| --- | --- | --- |
| `sigma` | `1.0` | Initial regularization; doubled after a rejected candidate |
| `epsilon` | `1e-10` | Proximal-step and sufficient-decrease parameter |
| `alpha` | `0.1` | Sufficient-decrease coefficient |
| `max_subproblem_iter` | `200` | Maximum candidate attempts per outer iteration |
| `lambda_factor` | `0.1` | Factor for finite-difference step selection |
| `lambda_min` | `1e-12` | Minimum finite-difference step |
| `decrease_rule` | `:proximal_step` | Decrease required in the acceptance test (below) |
| `curvature` | `:bfgs` | `:bfgs`: identity, then safeguarded BFGS after each accepted step; `:zero`: ``B_j=0`` in every iteration, without update |

With ``T=\max_j\operatorname{tr}(B_j)`` for the curvature matrices of the current
subproblem, a candidate passes the decrease test when all objectives satisfy

```math
F_j(p)\le F_j(x)-\frac{\alpha}{2}\sigma\lVert p-x\rVert_2^2
\quad(\texttt{decrease\_rule=:proximal\_step}),
\qquad
F_j(p)\le F_j(x)-\frac{\alpha\varepsilon^2}{27\sigma(1+T)^2}
\quad(\texttt{:epsilon\_trace}),
```

and the proximal criterion is ``\sigma\lVert p-x\rVert_2<\varepsilon/(3(1+T))``.
A step norm equal to the threshold proceeds to the decrease test. With
`curvature=:zero`, ``T=0`` and the threshold is ``\varepsilon/3``.

The proximal criterion is the only convergence test and returns
`:proximal_step_tolerance`. That return has `abs_theta=NaN`, since the model measure was
not recomputed at the returned candidate. `abs_theta` and `norm_progress` are recorded
and never stop the method. Other exits are resource limits and failures.
`max_subproblem_iter` limits this method's inner attempts, not Ipopt iterations.

!!! compat "Breaking change"
    The model-measure and progress stopping options of PDFPM and the choice of
    thresholds were removed without deprecation; the thresholds above are the only ones.

!!! compat "Breaking change"
    The default decrease of the acceptance test is now ``(\alpha/2)\sigma\lVert p-x\rVert_2^2``.
    Runs that relied on the former default change; `decrease_rule=:epsilon_trace`
    restores it bit by bit.

## Diagnostics of the last step test

At any `store_level`, the result keeps the data of the last step test. They are
only recorded and do not change the iterates.

| Field | Value |
| --- | --- |
| `sigma`, `max_tr_B` | σ and ``T`` of the test |
| `step_measure` | ``\sigma\lVert p-x\rVert_2`` |
| `step_threshold` | ``\varepsilon/(3(1+T))`` |
| `required_decrease` | The decrease of `decrease_rule` in the test |
| `actual_decrease` | ``\min_j[F_j(x)-F_j(p)]``; `NaN` if the test stopped at Step 3 |
| `n_sigma_doublings` | Total doublings of σ in the run |

All are `NaN` (and the count `0`) before the first test and for CondG and ProxGrad.

## Traces of σ and T

With any `store_level` other than `:none`, entry ``k`` of `sigma_trace` and `max_tr_B_trace` belongs
to entry ``k`` of the other traces (`x_trace` only from `:medium` on):

| Entry | `sigma_trace` | `max_tr_B_trace` |
| --- | --- | --- |
| Initial point | `NaN` | `NaN` |
| Accepted step | σ of the decrease test that accepted it | ``T`` of the subproblem that produced it |
| Proximal-step exit | σ of the stopping test | ``T`` of the subproblem that produced the candidate |

Here ``T=\max_j\operatorname{tr}(B_j)`` is taken before the BFGS update that follows
an accepted step. No test precedes the initial point, so both values are `NaN`
there. A run that stops on a rejected candidate or a failure records no new entry.

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
