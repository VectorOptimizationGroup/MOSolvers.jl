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

The default `stop_criteria=:proxgrad` stops when `abs_theta <= opt_tol`.
`:NormProgress` requires both that inequality and `norm_progress <= ftol`
after an iteration. The initial point is checked using `abs_theta` alone;
an exactly zero model measure also stops before another line search.

See [Configuration](@ref), [Line search](@ref backtracking), and
[Results and stopping](@ref) for common behavior.
