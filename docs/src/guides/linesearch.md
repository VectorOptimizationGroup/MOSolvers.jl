# [Line search](@id backtracking)

CondG and ProxGrad use
[`BackTracking`](https://vectoroptimizationgroup.github.io/MultiObjectiveLineSearches.jl/dev/backtracking/)
from [MultiObjectiveLineSearches.jl](https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl).
See the [package documentation](https://vectoroptimizationgroup.github.io/MultiObjectiveLineSearches.jl/dev/)
for its methods and API.

Pass its configuration through the `linesearch` keyword:

```jldoctest linesearch
julia> using MOSolvers

julia> search = BackTracking(c_1=1e-4, initial_step=InitialStatic(alpha=0.5));

julia> f(x) = [x[1]^2, (x[1] - 2)^2];

julia> Jf(x) = reshape([2x[1], 2(x[1] - 2)], 2, 1);

julia> result = CondG(f, Jf, [ones(1, 1), ones(1, 1)], 0.0, [3.0],
           CondG_options(verbose=0); lb=[-3.0], ub=[3.0], linesearch=search);

julia> converged(result)
true
```

The search follows ``d=p-x`` with an initial step and `maxstep` at most one.
The Armijo condition is componentwise:

```math
F_j(x+\alpha d)\le F_j(x)-c_1\alpha\lvert\theta\rvert,
\qquad j=1,\ldots,m.
```

The interpolation slopes use the smooth Jacobian product ``Jf(x)d``. The
search is configured once per solver call in the element type of `x0`.

`BackTracking` controls `c_1`, interpolation safeguards `tau_1`/`tau_2`,
`minstep`, `maxstep`, `max_iters`, and its own `verbose` level. Here `max_iters`
counts step reductions; the initial evaluation is additional. Nonfinite trials
are reduced by halving the step. An evaluator failure ends the search.

PDFPM selects candidates through its own regularization and decrease loop;
its controls are described on the [PDFPM](@ref) page.
