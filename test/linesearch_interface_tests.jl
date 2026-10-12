using MOSolvers
using MultiObjectiveLineSearches
using LinearAlgebra
using Test

@testset "Line-search configuration of CondG and ProxGrad" begin
    @test MOSolvers.BackTracking === MultiObjectiveLineSearches.BackTracking
    @test MOSolvers.InitialStatic === MultiObjectiveLineSearches.InitialStatic
    spec = BackTracking(c_1=0.1, initial_step=InitialStatic(alpha=0.5))
    for T in (Float32, Float64)
        ls = MOSolvers.configure_linesearch(spec, T)
        @test ls == configure_linesearch(spec, T)
        @test ls.c_1 isa T
    end
    # Steps above one leave the segment [x, p].
    @test_throws ArgumentError MOSolvers.configure_linesearch(BackTracking(maxstep=1.5), Float64)
    @test_throws ArgumentError MOSolvers.configure_linesearch(
        BackTracking(maxstep=2, initial_step=InitialStatic(alpha=1.5)), Float64)
    short = BackTracking(maxstep=0.5, initial_step=InitialStatic(alpha=0.5))
    @test MOSolvers.configure_linesearch(short, Float64).maxstep == 0.5
    @test_throws ArgumentError MOSolvers.configure_linesearch(BackTracking(c_1=2), Float64)
    # Valid in Float64, but minstep underflows to zero in Float32.
    tiny = BackTracking(minstep=1e-50)
    @test MOSolvers.configure_linesearch(tiny, Float64).minstep == 1e-50
    @test_throws ArgumentError MOSolvers.configure_linesearch(tiny, Float32)
    @test_throws ArgumentError MOSolvers.configure_linesearch(MOSolvers.ArmijoParams(), Float64)

    # The solvers reject the line search before evaluating anything.
    calls = Ref(0)
    evalf(x) = (calls[] += 1; [x[1]^2 / 2])
    evalJf(x) = (calls[] += 1; reshape([x[1]], 1, 1))
    problem = (evalf, evalJf, [ones(1, 1)], 0.0, [2.0])
    bounds = (; lb=[-3.0], ub=[3.0])
    bad = BackTracking(maxstep=2)
    @test_throws ArgumentError CondG(problem..., CondG_options(verbose=0); bounds..., linesearch=bad)
    @test_throws ArgumentError ProxGrad(problem..., ProxGrad_options(verbose=0); bounds..., linesearch=bad)
    @test calls[] == 0
    @test CondG(problem..., CondG_options(verbose=0, max_iter=0); bounds..., linesearch=spec).iter == 0
    @test ProxGrad(problem..., ProxGrad_options(verbose=0, max_iter=0); bounds..., linesearch=spec).iter == 0
end

@testset "Line search of CondG and ProxGrad ($T)" for T in (Float32, Float64)
    A = T[1 0 0 0; 0 2 0 0; 0 0 3 1]
    f(y) = T[sum(abs2, y) / 2 + A[1, :]' * y, sum(abs2, y .- 1), (A[3, :]' * y)^2 + y[2]^4]
    x = T[0.5, -0.25, 1, 0.75]
    JF = vcat((x .+ A[1, :])', (2 .* (x .- 1))', (2 * (A[3, :]' * x) .* A[3, :] .+ T[0, 4x[2]^3, 0, 0])')
    Fx = f(x)
    theta = T(-0.8)
    trials = Vector{T}[]
    F(y) = (push!(trials, copy(y)); (f(y), nothing))
    F!(Fy, y) = (push!(trials, copy(y)); Fy .= f(y); nothing)
    ls = MOSolvers.configure_linesearch(BackTracking(), T)
    x_trial, F_trial, slopes = similar(x), similar(Fx), similar(Fx)
    inputs = map(copy, (x, Fx, JF))

    # The same trials as armijo, with and without descent for every objective.
    for p in (T[0, 1, 0, 0.5], T[-0.5, 0.25, 0, 0])
        d = p .- x
        old = MOSolvers.armijo(F, x, d, Fx, JF; factor=-abs(theta))
        native = copy(trials); empty!(trials)
        new = MOSolvers.perform_linesearch!(ls, F!, x, JF, d, theta, Fx, x_trial, F_trial, slopes)
        @test isequal(trials, native); empty!(trials)
        @test slopes == [dot(view(JF, i, :), d) for i in 1:3]
        @test new.success == old.success && new.nfev == old.nfev
        if new.success
            @test new.alpha === old.stp
            @test F_trial == old.F_final && x_trial == x .+ new.alpha .* d
        end
        @test (x, Fx, JF) == inputs
    end

    # A failure reported by the evaluator ends the search with its message.
    d = T[0, 1, 0, 0.5] .- x
    failing!(Fy, y) = "evalh! failed: backend error"
    r = MOSolvers.perform_linesearch!(ls, failing!, x, JF, d, theta, Fx, x_trial, F_trial, slopes)
    @test !r.success && r.reason === :evaluation_failure
    @test r.message == "evalh! failed: backend error" && r.nfev == 1
    quiet!(Fy, y) = (Fy .= f(y); nothing)
    @test (@inferred MOSolvers.perform_linesearch!(ls, quiet!, x, JF, d, theta, Fx, x_trial, F_trial, slopes)).success
end

@testset "CondG and ProxGrad in Float32 with real backends" begin
    # ProxGrad's subproblem returns θ in Float64; the search factor is converted to T.
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5)^2 + (x[2] - 5)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5) 2(x[2] - 5)]
    A = [Matrix{Float32}(I, 2, 2) for _ in 1:2]
    x0, bounds = Float32[1000, 2000], (; lb=fill(-500f0, 2), ub=fill(10000f0, 2))
    kw = (; verbose=0, max_iter=50, opt_tol=1e-6, ftol=1e-6)
    for result in (CondG(evalf, evalJf, A, 0.0, x0, CondG_options(; kw...); bounds...),
                   ProxGrad(evalf, evalJf, A, 0.0, x0, ProxGrad_options(; kw...); bounds...))
        @test result.success
        @test eltype(result.x) === Float32 && eltype(result.Fval) === Float32
        @test all(isfinite, result.Fval)
    end
end
