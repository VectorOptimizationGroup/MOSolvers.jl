# Quick start

Consider two smooth objectives on a bounded interval:

```math
\min_{-3\le x\le 3} \bigl(x^2,\;(x-2)^2\bigr).
```

Define the objective vector and its Jacobian, whose rows are the objective
gradients. Set `delta=0` for the smooth case; `A` contains one matrix per objective.

```jldoctest quickstart
julia> using MOSolvers

julia> f(x) = [x[1]^2, (x[1] - 2)^2];

julia> Jf(x) = reshape([2x[1], 2(x[1] - 2)], 2, 1);

julia> A = [ones(1, 1), ones(1, 1)];

julia> x0 = [3.0];

julia> bounds = (; lb=[-3.0], ub=[3.0]);

julia> result = ProxGrad(f, Jf, A, 0.0, x0, ProxGrad_options(verbose=0); bounds...);

julia> converged(result)
true

julia> (length(result.x), length(result.Fval), x0 == [3.0])
(1, 2, true)

julia> all(isapprox.(result.Fval, f(result.x); atol=1e-8))
true
```

The result is one returned point and its objective vector. Read `result.status`
for the stopping cause and `result.message` for an explanation.

The same data can be passed to CondG or PDFPM:

```jldoctest quickstart
julia> cg = CondG(f, Jf, A, 0.0, x0, CondG_options(verbose=0); bounds...);

julia> pd = PDFPM(f, A, 0.0, x0, PDFPM_options(verbose=0); evalJf=Jf, bounds...);

julia> (converged(cg), converged(pd))
(true, true)
```

See [Problem formulation](@ref) for a composite example and
[Results and stopping](@ref) for the meaning of convergence.
