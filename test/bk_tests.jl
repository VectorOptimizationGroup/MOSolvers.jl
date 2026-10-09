using MOSolvers
using LinearAlgebra
using Random
using Test

const ZERO_DELTA = 0.0

identity_data(n::Int, m::Int) = ([Matrix{Float64}(I, n, n) for _ in 1:m], ZERO_DELTA)

const _BK_RNG = MersenneTwister(7)

@inline function bk1_problem()
    lb = fill(-500.0, 2)
    ub = fill(10000.0, 2)
    λ = rand(_BK_RNG)
    x0 = λ .* lb .+ (1 - λ) .* ub

    evalf(x) = [x[1]^2 + x[2]^2,
                (x[1] - 5.0)^2 + (x[2] - 5.0)^2]

    function evalJf(x)
        J = zeros(Float64, 2, 2)
        J[1, 1] = 2.0 * x[1]
        J[1, 2] = 2.0 * x[2]
        J[2, 1] = 2.0 * (x[1] - 5.0)
        J[2, 2] = 2.0 * (x[2] - 5.0)
        return J
    end

    return (; evalf, evalJf, lb, ub, x0)
end

@testset "BK1 with CondG" begin
    prob = bk1_problem()
    A, delta = identity_data(length(prob.x0), length(prob.evalf(prob.x0)))
    opts = CondG_options(verbose=0, max_iter=50, opt_tol=1e-6, ftol=1e-6, max_time=5.0)

    result = CondG(prob.evalf, prob.evalJf, A, delta, copy(prob.x0), opts; lb=prob.lb, ub=prob.ub)

    @test result.success
    @test all(isfinite, result.Fval)
    @test all(prob.lb .- 1e-8 .<= result.x .<= prob.ub .+ 1e-8)
end

@testset "BK1 with ProxGrad" begin
    prob = bk1_problem()
    A, delta = identity_data(length(prob.x0), length(prob.evalf(prob.x0)))
    opts = ProxGrad_options(verbose=0, max_iter=50, opt_tol=1e-6, ftol=1e-6, max_time=5.0, mu=1.0)

    result = ProxGrad(prob.evalf, prob.evalJf, A, delta, copy(prob.x0), opts; lb=prob.lb, ub=prob.ub)

    @test result.success
    @test all(isfinite, result.Fval)
    @test all(prob.lb .- 1e-8 .<= result.x .<= prob.ub .+ 1e-8)
end

@testset "BK1 with PDFPM" begin
    prob = bk1_problem()
    A, delta = identity_data(length(prob.x0), length(prob.evalf(prob.x0)))
    opts = PDFPM_options(verbose=0, max_iter=50, opt_tol=1e-6, ftol=1e-6, max_time=5.0, sigma=1.0, alpha=0.1, max_subproblem_iter=200)

    result = PDFPM(prob.evalf, A, delta, copy(prob.x0), opts; evalJf=prob.evalJf, lb=prob.lb, ub=prob.ub)

    @test result.success
    @test all(isfinite, result.Fval)
    @test all(prob.lb .- 1e-8 .<= result.x .<= prob.ub .+ 1e-8)
end
