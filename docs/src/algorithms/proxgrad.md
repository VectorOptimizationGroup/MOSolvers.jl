# ProxGrad

`ProxGrad` implements a proximal-gradient method for composite multiobjective
optimization [Tanabe2019Proximal](@cite).

```julia
result = ProxGrad(evalf, evalJf, A, delta, x0,
                  ProxGrad_options(mu=1.0);
                  lb=lower, ub=upper, linesearch=BackTracking())
```

The method solves a proximal direction subproblem with Ipopt and searches
along ``d=p-x``. The positive parameter `mu` weights the squared Euclidean
displacement in the subproblem. Its model objective supplies `abs_theta`.
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
