using MOSolvers
using LinearAlgebra
using Test

# Thresholds of PDFPM's proximal-step stop and sufficient-decrease acceptance.
# Inputs are chosen so that every stop threshold is exact; equality cases for
# the decrease compare against the value the solver computes.

step_tolerances(rule, sigma, B; epsilon=1.0, alpha=0.5) =
    MOSolvers._pdfpm_step_tolerances(
        MOSolvers._convert_pdfpm_options(Float64,
            PDFPM_options(verbose=0, epsilon=epsilon, alpha=alpha, step_tolerance_rule=rule)),
        sigma, B)

# One-dimensional run with h = 0 and controlled subproblem candidates. The first
# call returns theta = -1 at x0 = 2; later calls return `p_values[call]` (or x)
# with theta = 0, so an accepted candidate ends the run by optimality.
function step_case(rule; p_values, trial_F=Dict{Float64,Float64}(), B0=3.0,
                   epsilon=1.0, alpha=0.5, max_subproblem_iter=200, verbose=0)
    result = MOSolvers.OptimResult([2.0], [0.0])
    f!(F, x) = (F[1] = get(trial_F, x[1], x[1]^2 / 2); F)
    Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1]; J)
    h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
    calls = Ref(0)
    function subproblem(x, J, h, method; sigma, B)
        calls[] += 1
        p = get(p_values, calls[], x[1])
        return (haskey(p_values, calls[]) || calls[] == 1 ? -1.0 : 0.0), [p], true, ""
    end
    options = MOSolvers._convert_pdfpm_options(Float64,
        PDFPM_options(; verbose, epsilon, alpha, sigma=1.0, opt_tol=1e-8,
                      max_subproblem_iter, step_tolerance_rule=rule))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                               (iter, result) -> false; initial_B=[fill(B0, 1, 1)])
    return result, calls[]
end

@testset "PDFPM step tolerance rule: options" begin
    @test PDFPM_options().step_tolerance_rule === :epsilon
    @test PDFPM_options(step_tolerance_rule=:trace_scaled).step_tolerance_rule === :trace_scaled
    @test_throws ArgumentError PDFPM_options(step_tolerance_rule=:v2)
    params = MOSolvers._convert_pdfpm_options(Float32, PDFPM_options(step_tolerance_rule=:trace_scaled))
    @test params.step_tolerance_rule === :trace_scaled
    @test ("Step Tol Rule" => "trace_scaled") in MOSolvers._solver_header_pairs(params)
end

@testset "PDFPM step tolerance rule: thresholds" begin
    # T = max_j tr(B_j) = max(1 + 2, 0.5 + 0.5) = 3, so 1 + T = 4.
    B = [Matrix(Diagonal([1.0, 2.0])), Matrix(Diagonal([0.5, 0.5]))]

    stop, decrease = step_tolerances(:epsilon, 2.0, B)
    @test decrease == 0.5 / (2 * 2.0)                 # αε²/(2σ)
    @test stop(1.0)                                   # σ‖p-x‖ = ε stops
    @test !stop(nextfloat(1.0))

    # ε = 3/4: ε/(3(1+T)) = 1/16 exactly.
    stop, decrease = step_tolerances(:trace_scaled, 2.0, B; epsilon=0.75)
    @test decrease == 0.5 * 0.75^2 / (27 * 2.0 * 16)  # αε²/(27σ(1+T)²)
    @test !stop(1 / 16)                               # σ‖p-x‖ = ε/(3(1+T)) proceeds
    @test stop(prevfloat(1 / 16))

    # B = 0 gives T = 0: thresholds ε/3 and αε²/(27σ); ε = 3/4 gives 1/4.
    stop, decrease = step_tolerances(:trace_scaled, 1.0, [zeros(1, 1)]; epsilon=0.75)
    @test decrease == 0.5 * 0.75^2 / 27
    @test !stop(0.25) && stop(prevfloat(0.25))
end

@testset "PDFPM step tolerance rule: inner loop" begin
    # B0 = 3 and ε = 3/4: 1 + T = 4, stop threshold ε/(3·4) = 1/16 and decrease
    # αε²/(27σ·16) with σ = 1.
    epsilon = 0.75
    for verbose in (0, 1)
        # Step norm equal to the trace-scaled threshold: :trace_scaled proceeds to
        # the decrease test and accepts; :epsilon stops at the candidate.
        p = 2.0 - 1 / 16
        result, calls = step_case(:trace_scaled; p_values=Dict(1 => p), epsilon, verbose)
        @test result.status === :optimality_tolerance
        @test result.x == [p] && result.iter == 1 && calls == 2
        result, calls = step_case(:epsilon; p_values=Dict(1 => p), epsilon, verbose)
        @test result.status === :proximal_step_tolerance
        @test result.x == [p] && calls == 1
        @test result.message == "PDFPM proximal step satisfies sigma*norm(p-x) <= epsilon"

        # Step norm just below the trace-scaled threshold stops at the candidate.
        p = nextfloat(2.0 - 1 / 16)
        result, _ = step_case(:trace_scaled; p_values=Dict(1 => p), epsilon, verbose)
        @test result.status === :proximal_step_tolerance
        @test result.x == [p] && isnan(result.abs_theta)
        @test occursin("epsilon/(3(1+max_j tr(B_j)))", result.message)
    end

    # Decrease equal to the trace-scaled requirement is accepted; one ulp less
    # is rejected. σ|p - x| = 3/2 > ε passes both stop tests; the :epsilon
    # requirement αε²/(2σ) = 9/64 rejects both.
    p = 0.5
    _, decrease = step_tolerances(:trace_scaled, 1.0, [fill(3.0, 1, 1)]; epsilon)
    for (F_p, accepted) in ((2.0 - decrease, true), (nextfloat(2.0 - decrease), false))
        result, _ = step_case(:trace_scaled; p_values=Dict(1 => p), epsilon,
                              trial_F=Dict(p => F_p), max_subproblem_iter=1)
        @test result.status === (accepted ? :optimality_tolerance : :max_subproblem_iter)
        @test result.x == (accepted ? [p] : [2.0])
        result, _ = step_case(:epsilon; p_values=Dict(1 => p), epsilon,
                              trial_F=Dict(p => F_p), max_subproblem_iter=1)
        @test result.status === :max_subproblem_iter
    end
end

@testset "PDFPM step tolerance rule: KNOW-008 counterexample" begin
    # f(x) = x²/2, h = 0, B = 0, x0 = ε/3, α = 1/2. The model with B = 0 is
    # solved in closed form: p = x - f'(x)/σ, θ = -f'(x)²/(2σ). The closed form
    # ignores B, which the BFGS update leaves nonfinite after the first step
    # (sᵀBs = 0); only k = 0, where B = 0 holds, is under test.
    for epsilon in (1.0, 1e-3)
        x0 = epsilon / 3
        runs = map((:epsilon, :trace_scaled)) do rule
            result = MOSolvers.OptimResult([x0], [0.0])
            f!(F, x) = (result.n_f_evals += 1; F[1] = x[1]^2 / 2; F)
            Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1]; J)
            h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
            Bs = Float64[]
            function subproblem(x, J, h, method; sigma, B)
                push!(Bs, B[1][1, 1])
                g = J[1, 1]
                return -g^2 / (2 * sigma), [x[1] - g / sigma], true, ""
            end
            options = MOSolvers._convert_pdfpm_options(Float64,
                PDFPM_options(verbose=0, epsilon=epsilon, alpha=0.5, sigma=1.0,
                              opt_tol=0.0, max_iter=5, max_subproblem_iter=50,
                              step_tolerance_rule=rule))
            MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                                       (iter, result) -> false; initial_B=[zeros(1, 1)])
            (; result, Bs)
        end
        v1, v2 = runs
        # V1: σ|p - x| = ε/3 ≤ ε stops at k = 0.
        @test v1.result.status === :proximal_step_tolerance
        @test v1.result.iter == 1 && v1.Bs == [0.0]
        # V2: σ|p - x| = ε/3 equals ε/(3(1+T)) and proceeds; F(p) = 0 ≤
        # ε²/18 - ε²/54 accepts at the first attempt (σ = 1), then θ = 0 at
        # x = 0 stops by optimality.
        @test v2.result.status === :optimality_tolerance
        @test v2.result.iter == 1 && v2.result.x == [0.0]
        @test v2.Bs[1] == 0.0 && length(v2.Bs) == 2
        @test v2.result.n_f_evals == 2
    end
end

@testset "PDFPM step tolerance rule: default is :epsilon" begin
    # Bitwise equality with b190da0 was checked once on BK1/ZDT (see the
    # research memory); here the default must keep matching the explicit rule.
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    fields = (:x, :Fval, :iter, :abs_theta, :norm_progress, :n_f_evals, :n_Jf_evals,
              :x_trace, :Fval_trace, :abs_theta_trace, :norm_progress_trace, :status, :message)
    for jac in ((; evalJf), (;)), kw in ((;), (; epsilon=1e-2, alpha=0.5))
        runs = map(((;), (; step_tolerance_rule=:epsilon))) do rule
            options = PDFPM_options(; verbose=0, max_iter=50, opt_tol=1e-6,
                                    store_level=:medium, kw..., rule...)
            PDFPM(evalf, A, 0.0, [-140.0, 6850.0], options; jac...,
                  lb=fill(-500.0, 2), ub=fill(10000.0, 2))
        end
        @test all(isequal(getfield(runs[1], f), getfield(runs[2], f)) for f in fields)
    end
end
