# Changelog

## Unreleased

### Breaking

- `PDFPM`: the default decrease of the acceptance test (Step 4) is now
  `(α/2)σ‖p-x‖²`. Runs that relied on the former default change;
  `decrease_rule=:epsilon_trace` restores the former rule, `αε²/(27σ(1+T)²)`, bit by bit.

### Added

- `PDFPM_options(decrease_rule=:proximal_step | :epsilon_trace)`.
- `PDFPM_options(curvature=:bfgs | :zero)`: `:zero` keeps `Bⱼ = 0` in every iteration,
  without the BFGS update, so `T = 0` and the proximal-step threshold is `ε/3`.
  `:bfgs` (default) is the former behavior.

### Fixed

- CondG and ProxGrad use the current API of MultiObjectiveLineSearches (`main`, `d191add` or
  later). Installs resolve its `main` branch, whose API changed under the same version 0.1.0,
  so the former calls failed with a `MethodError` on the first line search.
- `PDFPM` without `evalJf`: the finite-difference step is
  `min(max(lambda_factor·ε/(σ√n), lambda_min), ε/(σ√n))`, so `lambda_min` never lifts it above
  the bound `ε/(σ√n)` of Step 1. The central difference costs `2n` evaluations of `f` (it no
  longer evaluates `f(x)`), and a perturbation that rounds to `x` stops the method with
  `:numerical_failure` instead of producing a zero derivative.
