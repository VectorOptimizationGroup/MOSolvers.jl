using MOSolvers
using LinearAlgebra
using Random
using Test

@testset "Steepest descent: weights of the direction" begin
    # KKT of min ½γ'Mγ over the simplex: with g = Mγ and λ = γ'Mγ, g ≥ λ, with equality where γ_i > 0.
    # m = 2 uses the closed form, 3 ≤ m ≤ 6 the faces, m ≥ 7 HiGHS.
    rng = MersenneTwister(20261128)
    for m in 1:8, n in (1, 10)
        G = randn(rng, m, n)
        m > 1 && (G[2, :] .= G[1, :])  # repeated gradient: singular face systems
        M = Symmetric(G * G')
        γ, success, _ = MOSolvers.simplex_qp(M)
        g, λ, tol = M * γ, dot(γ, M * γ), (m <= 6 ? 1e-10 : 1e-6) * (1 + norm(M))
        @test success && all(γ .>= 0) && sum(γ) ≈ 1
        @test all(g .>= λ - tol) && all(γ .* abs.(g .- λ) .<= tol)
    end
    γ, _, _ = MOSolvers.simplex_qp(Symmetric(Float32[2 1 0; 1 2 1; 0 1 2]))
    @test eltype(γ) === Float32
end

@testset "Steepest descent: direction and measures" begin
    rng = MersenneTwister(1)
    for m in (2, 3, 5), n in (1, 6)
        JF, d = randn(rng, m, n), zeros(n)
        MOSolvers.steepest_descent_direction!(d, JF)
        # Every objective decreases at least ‖d‖² along d (Fliege and Svaiter, Lemma 1).
        @test all(JF * d .<= -dot(d, d) + 1e-10)
        # ‖v‖_∞ ≤ ‖v‖₂ ≤ √n‖v‖_∞ carries over to the two measures.
        G∞ = MOSolvers.max_norm_measure(JF)[1]
        @test G∞ <= norm(d) + 1e-9 && norm(d) <= sqrt(n) * G∞ + 1e-9
    end
    # Opposite gradients: Pareto critical, so both measures vanish.
    JF, d = [1.0 2.0; -1.0 -2.0], ones(2)
    MOSolvers.steepest_descent_direction!(d, JF)
    @test norm(d) <= 1e-14 && MOSolvers.max_norm_measure(JF)[1] <= 1e-12
    # One objective: d = -∇f and the maximum-norm measure is ‖∇f‖_∞.
    MOSolvers.steepest_descent_direction!(d, [3.0 -4.0])
    @test d == [-3.0, 4.0] && MOSolvers.max_norm_measure([3.0 -4.0])[1] == 4.0
    # Two objectives: the linear program against a fine grid of the segment.
    JF = [1.0 -2.0 0.5; -0.5 1.0 1.5]
    grid = minimum(t -> norm(JF' * [t, 1 - t], Inf), range(0, 1; length=100_001))
    @test MOSolvers.max_norm_measure(JF)[1] ≈ grid atol = 1e-4
end
