using MOSolvers
using LinearAlgebra
using Random
using Test

# hⱼ(x) = δ‖Ãⱼ⁻ᵀx‖₁ against the linear program and a BigFloat reference, its
# validation, and the linear-programming path of the four-argument constructor.

free_bounds(n) = (fill(-Inf, n), fill(Inf, n))

function lp_params(A, delta)
    n = size(A[1], 1)
    return SubProblemParams([[Aj; -Aj] for Aj in A], [fill(delta, 2n) for _ in A], free_bounds(n)...)
end

closed_params(A, delta; kw...) = robust_subproblem_params(A, delta, free_bounds(size(A[1], 1))...; kw...)

evaluate(params, x, m) = (h = zeros(eltype(x), m); (evalh!(h, x, params)..., h))

# Bound of the closed form against the exact value: c κ u ‖c‖₁ δ with this c.
const EVALH_ERROR_CONSTANT = 2.0

@testset "Closed form of h against the linear program" begin
    for n in (2, 5, 10), delta in (0.01, 0.1, 1.0), seed in 1:5
        rng = MersenneTwister(100n + seed)
        A = [rand(rng, n, n) + n * I for _ in 1:2]
        x = randn(rng, n)
        ok_lp, _, h_lp = evaluate(lp_params(A, delta), x, 2)
        ok, _, h = evaluate(closed_params(A, delta), x, 2)
        @test ok && ok_lp
        @test isapprox(h, h_lp; atol=1e-6)
    end
    clear_evalh_cache!()
end

@testset "Closed form of h against a BigFloat reference" begin
    for n in (2, 5, 10, 30, 100), seed in 1:20
        rng = MersenneTwister(1000n + seed)
        A = [rand(rng, n, n)]
        kappa = h_matrix_diagnostics(A)[1].kappa
        kappa <= 1e8 || continue
        x = randn(rng, n)
        delta = 0.1
        ok, _, h = evaluate(closed_params(A, delta), x, 1)
        c = big.(A[1])' \ big.(x)
        exact = big(delta) * norm(c, 1)
        @test ok
        @test abs(h[1] - exact) <= EVALH_ERROR_CONSTANT * kappa * eps(Float64) * norm(c, 1) * delta
    end
end

@testset "Fold of h: exact where the linear program errs" begin
    # c₁ ≈ -5e-8 lies below the optimality tolerance of HiGHS, which keeps the wrong
    # sign of that component: error -2δ|c₁| (INV-003-explicacao, §2).
    A = [[0.8 0.3; 0.2 0.9]]
    c = [-4.786e-8, 0.1186]
    x = A[1]' * c
    delta = 0.1
    exact = big(delta) * norm(big.(A[1])' \ big.(x), 1)
    _, _, h = evaluate(closed_params(A, delta), x, 1)
    @test abs(h[1] - exact) <= 4 * eps(Float64)
    clear_evalh_cache!()
end

@testset "Solvers with delta > 0 do not touch the evalh! model cache" begin
    evalf(x) = [sum(abs2, x), sum(abs2, x .- 1)]
    evalJf(x) = vcat(permutedims(2x), permutedims(2(x .- 1)))
    A = [[1.0 0.2; 0.1 1.0], [0.9 0.3; 0.2 1.1]]
    x0 = [2.0, -1.5]
    bounds = (; lb=fill(-5.0, 2), ub=fill(5.0, 2))
    clear_evalh_cache!()
    @test isempty(MOSolvers._EVALH_MODEL_CACHE)
    runs = (PDFPM(evalf, A, 0.1, x0, PDFPM_options(verbose=0, max_iter=5, epsilon=1e-4); evalJf, bounds...),
            PDFPM(evalf, A, 0.1, x0, PDFPM_options(verbose=0, max_iter=3, epsilon=1e-4); bounds...),
            ProxGrad(evalf, evalJf, A, 0.1, x0, ProxGrad_options(verbose=0, max_iter=5); bounds...),
            CondG(evalf, evalJf, A, 0.1, x0, CondG_options(verbose=0, max_iter=5); bounds...))
    @test all(r -> r.iter >= 1, runs)
    @test all(r -> r.status !== :evaluation_failure, runs)
    @test isempty(MOSolvers._EVALH_MODEL_CACHE)
    for r in runs
        h = zeros(2)
        evalh!(h, r.x, closed_params(A, 0.1))
        @test r.Fval ≈ evalf(r.x) + h
    end
end

@testset "Validation of the matrices" begin
    singular = [1.0 1.0; 1.0 1.0]
    ill = [1.0 1.0; 1.0 1.0 + 1e-10]
    good = [1.0 0.0; 0.0 2.0]
    @test_throws r"A\[2\] is singular.*κ₂ = Inf.*max_cond_A" closed_params([good, singular], 0.1)
    @test_throws r"A\[2\] is ill conditioned.*κ₂ = .*max_cond_A` = 1.0e8" closed_params([good, ill], 0.1)
    @test_throws ArgumentError closed_params([good, ill], 0.1)
    @test_throws r"A\[1\] has nonfinite" closed_params([[NaN 0.0; 0.0 1.0]], 0.1)
    @test_throws r"A\[1\] has nonfinite" closed_params([[Inf 0.0; 0.0 1.0]], 0.1)
    @test_throws r"A\[1\] must be square" robust_subproblem_params([ones(2, 3)], 0.1, free_bounds(3)...)
    @test_throws ArgumentError closed_params([good], -0.1)
    @test_throws ArgumentError closed_params([good], NaN)

    # The limit is configurable; κ₂(good) = 2.
    @test closed_params([good], 0.1; max_cond_A=2.5).h_factors !== nothing
    @test_throws r"κ₂ = 2.0.*max_cond_A` = 1.5" closed_params([good], 0.1; max_cond_A=1.5)
    @test_throws ArgumentError closed_params([good], 0.1; max_cond_A=0.0)
    @test_throws ArgumentError closed_params([good], 0.1; max_cond_A=NaN)
    # An infinite limit still rejects a singular matrix.
    @test_throws r"singular" closed_params([singular], 0.1; max_cond_A=Inf)

    diagnostics = h_matrix_diagnostics([good, singular, [NaN 0.0; 0.0 1.0]])
    @test diagnostics[1].kappa ≈ cond(good) && diagnostics[1].sigma_min ≈ 1.0
    @test diagnostics[2].kappa == Inf && diagnostics[2].sigma_min <= eps()
    @test isnan(diagnostics[3].kappa) && isnan(diagnostics[3].sigma_min)
    @test h_matrix_diagnostics([zeros(2, 2)])[1] == (; kappa=Inf, sigma_min=0.0)

    # Through the solvers and their option.
    evalf(x) = [sum(abs2, x) / 2]
    evalJf(x) = permutedims(x)
    start(A) = fill(2.0, size(A[1], 1))
    for (solve, opts) in ((CondG_options, (o, A, d) -> CondG(evalf, evalJf, A, d, start(A), o)),
                          (ProxGrad_options, (o, A, d) -> ProxGrad(evalf, evalJf, A, d, start(A), o)),
                          (PDFPM_options, (o, A, d) -> PDFPM(evalf, A, d, start(A), o; evalJf)))
        @test solve().max_cond_A == 1e8
        @test_throws ArgumentError solve(max_cond_A=-1.0)
        @test_throws r"A\[1\] is singular" opts(solve(verbose=0), [zeros(1, 1)], 0.1)
        @test_throws r"A\[1\] is ill conditioned" opts(solve(verbose=0, max_iter=0), [[1.0 1.0; 1.0 1.0 + 1e-10]], 0.1)
        @test_throws r"A\[1\] is ill conditioned" opts(solve(verbose=0, max_iter=0, max_cond_A=1.0),
                                                       [[1.0 0.0; 0.0 2.0]], 0.1)
    end
end

@testset "delta = 0: no factorization and no validation" begin
    singular = [1.0 1.0; 1.0 1.0]
    for A in ([singular], [[NaN 0.0; 0.0 1.0]])
        params = closed_params(A, 0.0)
        @test params.h_factors === nothing && params.delta == 0.0
        ok, _, h = evaluate(params, [1.0, -2.0], 1)
        @test ok && h == [0.0]
    end
    @test isempty(MOSolvers._EVALH_MODEL_CACHE)
    evalf(x) = [sum(abs2, x)]
    evalJf(x) = permutedims(2x)
    r = ProxGrad(evalf, evalJf, [singular], 0.0, [1.0, 1.0], ProxGrad_options(verbose=0, max_iter=2))
    @test r.status !== :evaluation_failure
end

@testset "Four-argument SubProblemParams keeps the linear-programming path" begin
    @test fieldtype(SubProblemParams{Float64}, :h_factors) ===
          Union{Nothing, Vector{LU{Float64, Matrix{Float64}, Vector{Int}}}}
    A = [[0.8 0.3; 0.2 0.9]]
    params = lp_params(A, 0.1)
    @test params.h_factors === nothing && isnan(params.delta)
    clear_evalh_cache!()
    ok, message, h = evaluate(params, [0.3, -0.7], 1)
    @test ok && message == "All subproblems solved successfully"
    @test !isempty(MOSolvers._EVALH_MODEL_CACHE)
    @test h[1] ≈ 0.1 * norm(A[1]' \ [0.3, -0.7], 1) atol = 1e-8
    clear_evalh_cache!()

    # General polyhedra and the zero-b convention are unchanged.
    box = SubProblemParams([[1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]], [[1.0, 2.0, 3.0, 4.0]], free_bounds(2)...)
    ok, _, h = evaluate(box, [1.0, -1.0], 1)
    @test ok && h[1] ≈ 5.0
    clear_evalh_cache!()
    zero_b = SubProblemParams(lp_params(A, 0.0).A, lp_params(A, 0.0).b, free_bounds(2)...)
    ok, _, h = evaluate(zero_b, [1.0, 1.0], 1)
    @test ok && h == [0.0]
    @test isempty(MOSolvers._EVALH_MODEL_CACHE)

    # Rank-deficient A keeps reporting the status of HiGHS.
    unbounded = SubProblemParams([[0.0; -0.0;;]], [[1.0, 1.0]], free_bounds(1)...)
    ok, message, _ = @test_logs (:warn, r"could not be solved") evaluate(unbounded, [1.0], 1)
    @test !ok && occursin("h_1", message) && occursin("HiGHS returned", message)
    clear_evalh_cache!()
end

@testset "evalh! closed form: failures, types and allocations" begin
    A = [[1.0 0.0; 0.0 2.0]]
    params = closed_params(A, 0.1)
    for bad in ([NaN, 1.0], [Inf, 1.0])
        ok, message, _ = evaluate(params, bad, 1)
        @test !ok && occursin("h_1", message) && occursin("nonfinite", message)
    end
    big_h = closed_params([[1e-300 0.0; 0.0 1e-300]], 1.0)
    ok, message, _ = evaluate(big_h, [1e300, 1e300], 1)
    @test !ok && occursin("h_1", message) && occursin("not finite", message)

    x = [1.0, -2.0]
    h = zeros(1)
    @test (@inferred evalh!(h, x, params)) isa Tuple{Bool, String}
    @test h == [0.1 * (1.0 + 1.0)]
    @test (evalh!(h, view([0.0, 1.0, -2.0], 2:3), params); h == [0.2])

    A32 = [Float32[1 0; 0 2]]
    p32 = robust_subproblem_params(A32, 0.5f0, fill(-Inf32, 2), fill(Inf32, 2))
    h32 = zeros(Float32, 1)
    @test evalh!(h32, Float32[2, 4], p32)[1] && h32 == Float32[2]
    @test p32.h_factors isa Vector{LU{Float32, Matrix{Float32}, Vector{Int}}}

    # Only the work vector is allocated.
    for n in (2, 30, 100)
        p = closed_params([rand(MersenneTwister(n), n, n) + n * I], 0.1)
        y, g = randn(MersenneTwister(n), n), zeros(1)
        evalh!(g, y, p)
        @test (@allocated evalh!(g, y, p)) <= 8n + 128
    end
end
