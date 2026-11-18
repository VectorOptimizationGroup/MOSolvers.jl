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

The method stops when `abs_theta < opt_tol`, strictly, including at the initial
point. With `opt_tol=0` this test never holds, and an exactly zero model measure,
for which no line search exists, still stops before another search.
`norm_progress` is recorded and never stops the method.

!!! compat "Breaking change"
    The progress stopping options and their status were removed without deprecation.

See [Configuration](@ref), [Line search](@ref backtracking), and
[Results and stopping](@ref) for common behavior.
