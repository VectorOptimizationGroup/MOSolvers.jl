using MOSolvers
using Test

# Controlled subproblems isolate termination from backend tolerances. The real
# objective and line search still run; every proposed step halves x.
function termination_case(method; verbose=0, max_iter=1, converge_at=typemax(Int),
                          fail_at=0, fail_h=false, fail_search=false,
                          abort=false, max_time=Inf, norm_progress=false, ftol=1.0,
                          theta_values=Dict{Int,Float64}(), ascent=false,
                          linesearch=BackTracking(), fail_h_at=0,
                          p_values=Dict{Int,Float64}(), initial_value=nothing,
                          bad_jacobian=false, max_subproblem_iter=200,
                          epsilon=1e-10, analytic=true,
                          store_level=:none, trial_value=nothing,
                          throw_initial=false, throw_subproblem=false, throw_callback=false)
    result = MOSolvers.OptimResult([2.0], [0.0])
    calls = Ref(0)
    h_calls = Ref(0)
    callbacks = Int[]
    function f!(f, x)
        throw_initial && x[1] == 2.0 && error("injected initial exception")
        fail_search && x[1] != 2.0 && error("injected evaluation failure")
        f[1] = x[1] == 2.0 && initial_value !== nothing ? initial_value :
               (ascent ? -1 : 1) * x[1]^2 / 2
        if trial_value !== nothing && x[1] == 1.0
            f[1] = trial_value
        end
    end
    Jf!(J, x; lambda=nothing) = (J[1, 1] = bad_jacobian ? NaN : x[1])
    function h!(h, x)
        h_calls[] += 1
        fill!(h, 0.0)
        if fail_h || h_calls[] == fail_h_at
            return false, "injected h failure"
        end
        return true, "h evaluated"
    end
    function subproblem(x, J, h, algorithm; kwargs...)
        throw_subproblem && error("injected subproblem exception")
        calls[] += 1
        theta = get(theta_values, calls[], calls[] >= converge_at ? 0.0 : -1.0)
        p = [get(p_values, calls[], x[1] / 2)]
        return theta, p, calls[] != fail_at, "injected subproblem failure"
    end
    function callback(iter, result)
        throw_callback && error("injected callback exception")
        @test result.status === :running
        @test !converged(result)
        push!(callbacks, iter)
        return abort
    end
    criterion = norm_progress ? :NormProgress : method
    kwargs = (; verbose, max_iter, max_time, opt_tol=1e-8, ftol,
              stop_criteria=criterion, store_level)
    if method == :condg
        options = MOSolvers._convert_condg_options(Float64, CondG_options(; kwargs...))
        MOSolvers._CondG_optimize!(result, f!, Jf!, h!, 1, subproblem, options, callback; linesearch)
    elseif method == :proxgrad
        options = MOSolvers._convert_proxgrad_options(Float64, ProxGrad_options(; kwargs...))
        MOSolvers._ProxGrad_optimize!(result, f!, Jf!, h!, 1, subproblem, options, callback; linesearch)
    else
        options = MOSolvers._convert_pdfpm_options(Float64, PDFPM_options(; kwargs..., max_subproblem_iter, epsilon))
        MOSolvers._PDFPM_optimize!(result, f!, Jf!, analytic, h!, 1, subproblem, options, callback)
    end
    @test result.status ∉ (:not_started, :running)
    @test result.message != "Optimization not started"
    @test result.total_time >= result.subproblem_time >= 0
    return result, callbacks
end

@testset "Termination status of the three solvers" begin
    for method in (:condg, :proxgrad, :pdfpm), verbose in (0, 1)
        @testset "$method verbose=$verbose" begin
            for max_iter in (0, 1)
                result, _ = termination_case(method; verbose, max_iter)
                @test !converged(result)
                @test result.status === :max_iter
                @test result.message == "Maximum number of iterations reached $max_iter"
                @test result.iter == max_iter

                result, _ = termination_case(method; verbose, max_iter, converge_at=1)
                @test converged(result)
                @test result.status === :optimality_tolerance
                @test occursin("Solution found at initial point", result.message)
                @test result.iter == 0

                result, _ = termination_case(method; verbose, max_iter, fail_at=1)
                @test !converged(result)
                @test occursin("injected subproblem failure", result.message)
                @test result.status === :subproblem_failure
            end

            for norm_progress in (false, true)
                result, _ = termination_case(method; verbose, converge_at=2, norm_progress)
                @test converged(result)
                @test result.status === (norm_progress ? :optimality_and_progress_tolerance : :optimality_tolerance)
                @test occursin("The solution was found using", result.message)
                @test result.iter == 1
                @test result.x == [1.0]
                @test result.Fval == [0.5]
            end

            for max_iter in (1, 2)
                result, callbacks = termination_case(method; verbose, max_iter, abort=true)
                @test !converged(result)
                @test result.status === :callback
                @test result.message == "Interrupted by callback"
                @test result.iter == 1
                @test callbacks == [1]

                result, _ = termination_case(method; verbose, max_iter, fail_at=2)
                @test !converged(result)
                @test occursin("injected subproblem failure", result.message)
                @test result.status === :subproblem_failure
            end

            result, _ = termination_case(method; verbose, max_time=0.0)
            @test !converged(result)
            @test result.status === :max_time
            @test result.message == "Maximum time reached 0.0"

            result, _ = termination_case(method; verbose, fail_h=true)
            @test !converged(result)
            @test result.status === :evaluation_failure
            @test occursin("injected h failure", result.message)

            if method != :pdfpm
                result, _ = termination_case(method; verbose, fail_search=true)
                @test !converged(result)
                @test result.status === :evaluation_failure
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
        @test converged(results[1]) == converged(results[2]) == iszero(x0[1])
        @test results[1].status == results[2].status
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
        @test converged(result)
        @test result.message == "The solution was found with |θ_$name| = 0"
        @test result.iter == 1 && result.x == [1.0] && result.Fval == [0.5]

        # A nonfinite θ ends the run before the search.
        for bad in (NaN, -Inf)
            result, _ = termination_case(method; max_iter=3, theta_values=Dict(2 => bad))
            @test !converged(result)
            @test result.status === :numerical_failure
            @test occursin("theta=$bad", result.message)
            @test result.iter == 0 && result.x == [2.0]
        end

        # A search failure keeps the last accepted point and reports its diagnostic.
        result, _ = termination_case(method; max_iter=3, ascent=true,
                                     linesearch=BackTracking(max_iters=2))
        @test !converged(result)
        @test result.status === :line_search_failure
        @test result.message == "Line search failed: Maximum number of backtracking reductions reached"
        @test result.iter == 0 && result.x == [2.0] && result.Fval == [-2.0]

        # The configured search sets the accepted step and objective values.
        spec = BackTracking(initial_step=InitialStatic(alpha=0.5))
        result, _ = termination_case(method; linesearch=spec)
        @test result.iter == 1 && result.x == [1.5] && result.Fval == [1.125]

        # Nonfinite trials are recoverable and do not poison the solver status.
        result, _ = termination_case(method; trial_value=NaN)
        @test result.status === :max_iter
        @test result.x == [1.5] && result.Fval == [1.125]

        result, _ = termination_case(method; ascent=true, linesearch=BackTracking(minstep=0.75))
        @test result.status === :line_search_failure
        @test occursin("minimum", result.message)

        tiny = BackTracking(minstep=1e-20, initial_step=InitialStatic(alpha=1e-19))
        result, _ = termination_case(method; linesearch=tiny)
        @test result.status === :line_search_failure
        @test occursin("indistinguishable", result.message)
    end
end

@testset "Result contract and cause precedence" begin
    result = MOSolvers.OptimResult(Float32[1], Float32[1])
    @test result.status === :not_started
    @test !hasproperty(result, :success)
    @test !(@inferred converged(result))
    @test isnan(result.abs_theta)

    for method in (:condg, :proxgrad, :pdfpm), verbose in (0, 1)
        # A completed convergence check wins over callback and loop exhaustion.
        result, callbacks = termination_case(method; verbose, converge_at=2, abort=true)
        @test result.status === :optimality_tolerance
        @test converged(result) && isempty(callbacks)
        @test result.total_time >= 0
        @test result.message != "Optimization not started"

        # Preserve initialization policy: zero budget still checks the initial point.
        result, _ = termination_case(method; verbose, max_iter=0, max_time=0, converge_at=1)
        @test result.status === :optimality_tolerance
        result, _ = termination_case(method; verbose, max_iter=0, max_time=0)
        @test result.status === :max_iter

        # A failed next subproblem keeps the last fully recorded x/F checkpoint.
        result, _ = termination_case(method; verbose, fail_at=2)
        @test result.status === :subproblem_failure
        @test result.iter == 0 && result.x == [2.0] && result.Fval == [2.0]
        @test !converged(result)

        # Invalid model outputs cannot be called convergence, even at max_iter=0.
        for bad in (NaN, Inf, -Inf)
            result, _ = termination_case(method; verbose, max_iter=0, theta_values=Dict(1 => bad))
            @test result.status === :numerical_failure
            @test !converged(result)
            result, _ = termination_case(method; verbose, max_iter=0, p_values=Dict(1 => bad))
            @test result.status === :numerical_failure
        end
        result, _ = termination_case(method; verbose, initial_value=NaN)
        @test result.status === :numerical_failure
        result, _ = termination_case(method; verbose, bad_jacobian=true)
        @test result.status === :numerical_failure

        @test_throws ErrorException termination_case(method; throw_initial=true)
        @test_throws ErrorException termination_case(method; throw_subproblem=true)
        @test_throws ErrorException termination_case(method; throw_callback=true)

        # Reported h failure remains evaluation_failure, including trial evaluations.
        result, _ = termination_case(method; verbose, fail_h_at=2)
        @test result.status === :evaluation_failure
        @test occursin("injected h failure", result.message)
        @test result.x == [2.0] && result.Fval == [2.0]
    end
end

@testset "PDFPM inner-loop causes" begin
    for verbose in (0, 1)
        # The proximal criterion is separate from the uncomputed model measure at p.
        p = 2.0 + 1e-12
        result, callbacks = termination_case(:pdfpm; verbose, p_values=Dict(1 => p), store_level=:medium)
        @test result.status === :proximal_step_tolerance
        @test converged(result) && isempty(callbacks)
        @test result.iter == 1 && result.x == [p]
        @test result.Fval == [p^2 / 2]
        @test isnan(result.abs_theta)
        @test isnan(result.abs_theta_trace[end])
        @test result.x_trace[end] == result.x && result.Fval_trace[end] == result.Fval

        # A failed evaluation while preparing this return must win over convergence.
        result, _ = termination_case(:pdfpm; verbose, p_values=Dict(1 => p), fail_h_at=2)
        @test result.status === :evaluation_failure
        @test !converged(result)

        result, _ = termination_case(:pdfpm; verbose, ascent=true, max_subproblem_iter=1)
        @test result.status === :max_subproblem_iter
        @test !converged(result) && result.iter == 0
        @test occursin("sigma=", result.message)

        # Failure during sigma refinement is not reported as the outer iteration limit.
        for analytic in (false, true)
            result, _ = termination_case(:pdfpm; verbose, ascent=true, max_subproblem_iter=2, fail_at=2, analytic)
            @test result.status === :subproblem_failure
            @test occursin("injected subproblem failure", result.message)
        end
        result, _ = termination_case(:pdfpm; verbose, ascent=true, max_subproblem_iter=2,
                                     p_values=Dict(2 => NaN))
        @test result.status === :numerical_failure

        # A rejected trial, successful refinement and final optimality check.
        result, _ = termination_case(:pdfpm; verbose, max_subproblem_iter=2,
                                     p_values=Dict(1 => 3.0, 2 => 1.0), converge_at=3)
        @test result.status === :optimality_tolerance
        @test result.iter == 1 && result.x == [1.0] && result.Fval == [0.5]

        # Even -Inf cannot count as sufficient decrease; retry with a finite point.
        for bad in (NaN, -Inf)
            result, _ = termination_case(:pdfpm; verbose, max_subproblem_iter=2,
                                         trial_value=bad, p_values=Dict(2 => 1.5), converge_at=3)
            @test result.status === :optimality_tolerance
            @test result.x == [1.5] && result.Fval == [1.125]
        end
    end
end

@testset "Backend diagnostics and finite differences through the public API" begin
    evalf(x) = [x[1]^2 / 2]
    evalJf(x) = reshape([x[1]], 1, 1)
    for method in (:condg, :proxgrad, :pdfpm)
        # A singular A leaves the support problem unbounded; retain HiGHS' status.
        A = [zeros(1, 1)]
        result = if method === :condg
            CondG(evalf, evalJf, A, 1.0, [2.0], CondG_options(verbose=0))
        elseif method === :proxgrad
            ProxGrad(evalf, evalJf, A, 1.0, [2.0], ProxGrad_options(verbose=0))
        else
            PDFPM(evalf, A, 1.0, [2.0], PDFPM_options(verbose=0); evalJf)
        end
        @test result.status === :evaluation_failure
        @test occursin("h_1", result.message) && occursin("HiGHS returned", result.message)
        @test !converged(result) && all(isnan, result.Fval)
    end

    # The unbounded CondG subproblem must not become an outer infeasibility claim.
    result = CondG(evalf, evalJf, [ones(1, 1)], 0.0, [2.0], CondG_options(verbose=0))
    @test result.status === :subproblem_failure
    @test occursin("The subproblem of CondG failed:", result.message)
    @test !converged(result)

    for T in (Float32, Float64), x0 in (T[0], T[2])
        result = PDFPM(evalf, [ones(T, 1, 1)], zero(T), x0,
                       PDFPM_options(verbose=0, max_iter=0, opt_tol=1e-6,
                                     epsilon=0.1, lambda_factor=1.0, lambda_min=1e-4);
                       lb=T[-3], ub=T[3])
        @test result.status === (iszero(x0[1]) ? :optimality_tolerance : :max_iter)
        @test converged(result) == iszero(x0[1])
        @test result.n_f_evals >= 4 && result.n_Jf_evals == 1
        @test result.x == x0 && eltype(result.Fval) === T
    end
end
