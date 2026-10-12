using MOSolvers
using Test

# Controlled subproblems isolate termination from backend tolerances. The real
# objective and line search still run; every proposed step halves x.
function termination_case(method; verbose=0, max_iter=1, converge_at=typemax(Int),
                          fail_at=0, fail_h=false, fail_search=false,
                          abort=false, max_time=Inf, norm_progress=false, ftol=1.0,
                          theta_values=Dict{Int,Float64}(), ascent=false,
                          linesearch=BackTracking())
    result = MOSolvers.OptimResult([2.0], [0.0])
    calls = Ref(0)
    callbacks = Int[]
    function f!(f, x)
        fail_search && x[1] != 2.0 && error("injected evaluation failure")
        f[1] = (ascent ? -1 : 1) * x[1]^2 / 2
    end
    Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1])
    function h!(h, x)
        fill!(h, 0.0)
        if fail_h
            result.success = false
            result.message = "injected h failure"
        end
    end
    function subproblem(x, J, h, algorithm; kwargs...)
        calls[] += 1
        if calls[] == fail_at
            result.success = false
            result.message = "injected subproblem failure"
        end
        theta = get(theta_values, calls[], calls[] >= converge_at ? 0.0 : -1.0)
        return theta, x ./ 2
    end
    callback(iter, result) = (push!(callbacks, iter); abort)
    criterion = norm_progress ? :NormProgress : method
    kwargs = (; verbose, max_iter, max_time, opt_tol=1e-8, ftol,
              stop_criteria=criterion)
    if method == :condg
        options = MOSolvers._convert_condg_options(Float64, CondG_options(; kwargs...))
        MOSolvers._CondG_optimize!(result, f!, Jf!, h!, 1, subproblem, options, callback; linesearch)
    elseif method == :proxgrad
        options = MOSolvers._convert_proxgrad_options(Float64, ProxGrad_options(; kwargs...))
        MOSolvers._ProxGrad_optimize!(result, f!, Jf!, h!, 1, subproblem, options, callback; linesearch)
    else
        options = MOSolvers._convert_pdfpm_options(Float64, PDFPM_options(; kwargs...))
        MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options, callback)
    end
    return result, callbacks
end

@testset "Termination with current result fields" begin
    for method in (:condg, :proxgrad, :pdfpm), verbose in (0, 1)
        @testset "$method verbose=$verbose" begin
            for max_iter in (0, 1)
                result, _ = termination_case(method; verbose, max_iter)
                @test !result.success
                @test result.message == "Maximum number of iterations reached $max_iter"
                @test result.iter == max_iter

                result, _ = termination_case(method; verbose, max_iter, converge_at=1)
                @test result.success
                @test occursin("Solution found at initial point", result.message)
                @test result.iter == 0

                result, _ = termination_case(method; verbose, max_iter, fail_at=1)
                @test !result.success
                @test result.message == "injected subproblem failure"
            end

            for norm_progress in (false, true)
                result, _ = termination_case(method; verbose, converge_at=2, norm_progress)
                @test result.success
                @test occursin("The solution was found using", result.message)
                @test result.iter == 1
                @test result.x == [1.0]
                @test result.Fval == [0.5]
            end

            for max_iter in (1, 2)
                result, callbacks = termination_case(method; verbose, max_iter, abort=true)
                @test !result.success
                @test result.message == "Interrupted by callback"
                @test result.iter == 1
                @test callbacks == [1]

                result, _ = termination_case(method; verbose, max_iter, fail_at=2)
                @test !result.success
                @test result.message == "injected subproblem failure"
            end

            result, _ = termination_case(method; verbose, max_time=0.0)
            @test !result.success
            @test result.message == "Maximum time reached 0.0"

            result, _ = termination_case(method; verbose, fail_h=true)
            @test !result.success
            @test !occursin("Maximum number of iterations", result.message)

            if method != :pdfpm
                result, _ = termination_case(method; verbose, fail_search=true)
                @test !result.success
                @test occursin("injected evaluation failure", result.message)
            end
        end
    end
end

@testset "Public API termination with real backends" begin
    evalf(x) = [x[1]^2 / 2]
    evalJf(x) = reshape([x[1]], 1, 1)
    A = [ones(1, 1)]
    for method in (:condg, :proxgrad, :pdfpm), x0 in ([2.0], [0.0])
        results = map((0, 1)) do verbose
            kwargs = (; verbose, max_iter=0, opt_tol=1e-6)
            bounds = (; lb=[-3.0], ub=[3.0])
            if method == :condg
                CondG(evalf, evalJf, A, 0.0, x0, CondG_options(; kwargs...); bounds...)
            elseif method == :proxgrad
                ProxGrad(evalf, evalJf, A, 0.0, x0, ProxGrad_options(; kwargs...); bounds...)
            else
                PDFPM(evalf, A, 0.0, x0, PDFPM_options(; kwargs...); evalJf, bounds...)
            end
        end
        @test results[1].success == results[2].success == iszero(x0[1])
        @test results[1].message == results[2].message
        @test all(r -> r.iter == 0, results)
        @test x0 == results[1].x == results[2].x
        if !iszero(x0[1])
            @test results[1].message == "Maximum number of iterations reached 0"
        end
    end
end

@testset "Line search inside CondG and ProxGrad" begin
    for method in (:condg, :proxgrad)
        name = method == :condg ? "CondG" : "ProxGrad"
        # θ = 0 when norm_progress does not stop the run: x is critical, no search.
        result, _ = termination_case(method; max_iter=3, converge_at=2, norm_progress=true, ftol=0.1)
        @test result.success
        @test result.message == "The solution was found with |θ_$name| = 0"
        @test result.iter == 1 && result.x == [1.0] && result.Fval == [0.5]

        # A nonfinite θ ends the run before the search.
        for bad in (NaN, -Inf)
            result, _ = termination_case(method; max_iter=3, theta_values=Dict(2 => bad))
            @test !result.success
            @test result.message == "The subproblem of $name returned θ = $bad"
            @test result.iter == 1 && result.x == [1.0]
        end

        # A search failure keeps the last accepted point and reports its diagnostic.
        result, _ = termination_case(method; max_iter=3, ascent=true,
                                     linesearch=BackTracking(max_iters=2))
        @test !result.success
        @test result.message == "Line search failed: Maximum number of backtracking reductions reached"
        @test result.iter == 0 && result.x == [2.0] && result.Fval == [-2.0]

        # The configured search sets the accepted step and objective values.
        spec = BackTracking(initial_step=InitialStatic(alpha=0.5))
        result, _ = termination_case(method; linesearch=spec)
        @test result.iter == 1 && result.x == [1.5] && result.Fval == [1.125]
    end
end
