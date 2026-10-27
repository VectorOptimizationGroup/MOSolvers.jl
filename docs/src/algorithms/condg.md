# CondG

`CondG` implements a generalized conditional-gradient method for composite
multiobjective optimization [Assuncao2025Generalized](@cite).

```julia
result = CondG(evalf, evalJf, A, delta, x0, CondG_options();
               lb=lower, ub=upper, linesearch=BackTracking())
```

The method solves a direction subproblem with HiGHS and searches along
``d=p-x``. It uses the absolute subproblem model value as `abs_theta`.
Step selection uses `BackTracking` from
[MultiObjectiveLineSearches.jl](https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl).

The default `stop_criteria=:condg` stops when `abs_theta <= opt_tol`.
`:NormProgress` requires both that inequality and `norm_progress <= ftol`
after an iteration. The initial point is checked using `abs_theta` alone;
an exactly zero model measure also stops before another line search.

See [Configuration](@ref), [Line search](@ref backtracking), and
[Results and stopping](@ref) for common behavior.
