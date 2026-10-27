# Algorithms

| Method | Derivatives | Step control | Specific options |
| --- | --- | --- | --- |
| [CondG](@ref) | Analytical Jacobian | `linesearch` keyword | Common solver options |
| [ProxGrad](@ref) | Analytical Jacobian | `linesearch` keyword | `mu` |
| [PDFPM](@ref) | Analytical Jacobian or central differences | Internal regularization loop | `sigma`, `epsilon`, `alpha`, `max_subproblem_iter`, `lambda_factor`, `lambda_min`, `step_tolerance_rule` |

All three methods take one matrix per objective, `delta`, a floating-point
initial vector, and optional vector bounds. See [Problem formulation](@ref).
Each method returns the same result type; its model measure is method-specific.
