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

A candidate passes the decrease test when all objectives satisfy

```math
F_j(p)\le F_j(x)-\frac{\alpha\varepsilon^2}{2\sigma}.
```

The proximal criterion ``\sigma\lVert p-x\rVert_2\le\varepsilon`` returns
`:proximal_step_tolerance`. That return has `abs_theta=NaN`, since the model
measure was not recomputed at the returned candidate.

The default `stop_criteria=:pdfpm` also stops on `abs_theta <= opt_tol`.
`:NormProgress` combines that test with `norm_progress <= ftol` after an
iteration. The initial point is checked using `abs_theta` alone.
`max_subproblem_iter` limits this method's inner attempts, not Ipopt iterations.

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
