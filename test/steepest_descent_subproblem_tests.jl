using MOSolvers
using LinearAlgebra
using Random
using Test

# KKT conditions of min ½γ'Mγ over the simplex: with g = Mγ and λ = γ'Mγ,
# g_i ≥ λ for every i, and g_i = λ where γ_i > 0.
function simplex_kkt_residual(M, γ)
    g = M * γ
    λ = dot(γ, g)
    scale = 1 + maximum(abs, M)
    feasibility = max(maximum(max.(-γ, 0)), abs(sum(γ) - 1))
    dual = maximum(max.(λ .- g, 0))
    complementarity = maximum(γ .* abs.(g .- λ))
    return max(feasibility, dual / scale, complementarity / scale)
end

@testset "Steepest descent: QP over the simplex" begin
    rng = MersenneTwister(20261128)
    for m in 1:8, n in (1, 3, 10), trial in 1:5
        G = randn(rng, m, n)
        trial == 1 && m > 1 && (G[2, :] .= G[1, :])          # repeated gradient
        M = Symmetric(G * G')
        γ, success, message = MOSolvers.simplex_qp(M)
        @test success && message == ""
        @test simplex_kkt_residual(M, γ) <= (m <= 6 ? 1e-10 : 1e-6)
        if 3 <= m <= 6
            # The enumeration of the faces is not worse than HiGHS.
            γh, ok, _ = MOSolvers._simplex_qp_highs(M)
            @test ok && dot(γ, M * γ) <= dot(γh, M * γh) + 1e-9
        end
    end
    # Closed form for two objectives against the enumeration of the faces.
    for trial in 1:20
        G = randn(rng, 2, 4)
        M = Symmetric(G * G')
        @test MOSolvers._simplex_qp_two(M) ≈ MOSolvers._simplex_qp_faces(M) atol = 1e-12
    end
    γ32, ok, _ = MOSolvers.simplex_qp(Symmetric(Float32[2 1 0; 1 2 1; 0 1 2]))
    @test ok && eltype(γ32) === Float32 && sum(γ32) ≈ 1
end

@testset "Steepest descent: direction and θ" begin
    rng = MersenneTwister(1)
    for m in (1, 2, 3, 5), n in (2, 6)
        JF = randn(rng, m, n)
        d = zeros(n)
        γ, success, _ = MOSolvers.steepest_descent_direction!(d, JF)
        @test success
        @test d ≈ -JF' * γ
        # Every objective decreases at least ‖d‖² along d (Fliege and Svaiter, Lemma 1).
        @test all(JF * d .<= -dot(d, d) + 1e-10)
    end
    # Opposite gradients: the point is Pareto critical, d = 0 and both measures vanish.
    JF = [1.0 2.0; -1.0 -2.0]
    d = ones(2)
    MOSolvers.steepest_descent_direction!(d, JF)
    @test norm(d) <= 1e-14
    value, ok, _ = MOSolvers.max_norm_measure(JF)
    @test ok && value <= 1e-12
    # One objective: d = -∇f and the measure is ‖∇f‖_∞.
    JF = [3.0 -4.0]
    MOSolvers.steepest_descent_direction!(d, JF)
    @test d == [-3.0, 4.0]
    @test MOSolvers.max_norm_measure(JF)[1] == 4.0
end

@testset "Steepest descent: maximum-norm measure" begin
    rng = MersenneTwister(2)
    for m in (2, 3, 4), n in (1, 5), trial in 1:5
        JF = randn(rng, m, n)
        G∞, ok, _ = MOSolvers.max_norm_measure(JF)
        @test ok
        # No point of the simplex does better than the linear program.
        for _ in 1:200
            γ = -log.(rand(rng, m)); γ ./= sum(γ)
            @test G∞ <= norm(JF' * γ, Inf) + 1e-9
        end
        # ‖v‖_∞ ≤ ‖v‖₂ ≤ √n ‖v‖_∞ carries over to the two measures.
        d = zeros(n)
        MOSolvers.steepest_descent_direction!(d, JF)
        G₂ = norm(d)
        @test G∞ <= G₂ + 1e-9
        @test G₂ <= sqrt(n) * G∞ + 1e-9
    end
    # Two objectives: the linear program against a fine grid of the segment.
    JF = [1.0 -2.0 0.5; -0.5 1.0 1.5]
    grid = minimum(t -> norm(JF' * [t, 1 - t], Inf), range(0, 1; length=100_001))
    @test MOSolvers.max_norm_measure(JF)[1] ≈ grid atol = 1e-4
end
