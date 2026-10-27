# Problem formulation

The solvers use the componentwise composite objective

```math
\min_{x\in\Omega} F(x)=f(x)+h(x),\qquad
\Omega=\{x\in\mathbb R^n:\ell\le x\le u\},
```

where `evalf(x)` returns a vector of length ``m``. An analytical `evalJf(x)`
returns an ``m\times n`` matrix, with the gradient of objective ``j`` in row ``j``.

The nonsmooth term is specified by ``m`` square ``n\times n`` matrices and a
finite nonnegative scalar ``\delta``:

```math
h_j(x)=\max_{z\in Z_j}\langle x,z\rangle,\qquad
Z_j=\{z\in\mathbb R^n:-\delta\mathbf1\le A_jz\le\delta\mathbf1\}.
```

Nonsingular ``A_j`` give bounded sets ``Z_j``. The public solvers check matrix
sizes; singular matrices can lead to an unbounded support-function evaluation.
With `delta=0`, the solver evaluates `h` as zero. The matrices are still required
because their number determines the number of objectives.

For ``A_j=I``, the support term is ``h_j(x)=\delta\lVert x\rVert_1``. This gives
a small composite example:

```jldoctest composite
julia> using MOSolvers

julia> f(x) = [x[1]^2 / 2];

julia> Jf(x) = reshape([x[1]], 1, 1);

julia> result = ProxGrad(f, Jf, [ones(1, 1)], 0.1, [2.0],
           ProxGrad_options(verbose=0, max_iter=50);
           lb=[-3.0], ub=[3.0]);

julia> isapprox(result.Fval[1], f(result.x)[1] + 0.1abs(result.x[1]); atol=1e-7)
true

julia> converged(result)
true
```

Bounds are vectors of length ``n``; either side may be omitted for an infinite
bound. The solvers warn if `x0` lies outside the bounds. Supply a feasible
initial point, since directions are constructed from the current point and a
bounded subproblem solution.
