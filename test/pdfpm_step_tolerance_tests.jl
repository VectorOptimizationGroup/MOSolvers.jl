using MOSolvers
using LinearAlgebra
using Test

# Thresholds of PDFPM's proximal-step stop and sufficient-decrease acceptance
# (Step 3, constant 3; Step 4, (α/2)σ‖p-x‖² or the ε-trace rule with constant 27).
# Inputs are chosen so that every stop threshold is exact; equality cases for the
# decrease compare against the value the solver computes.

step_tolerances(sigma, B, step_norm=1.0; epsilon=1.0, alpha=0.5, decrease_rule=:proximal_step) =
    MOSolvers._pdfpm_step_tolerances(
        MOSolvers._convert_pdfpm_options(Float64,
            PDFPM_options(verbose=0, epsilon=epsilon, alpha=alpha, decrease_rule=decrease_rule)),
        sigma, B, step_norm)

# One-dimensional run with h = 0 and controlled subproblem candidates. The first
# call returns theta = -1 at x0 = 2; later calls return `p_values[call]` (or x)
# with theta = 0, so an accepted candidate ends the run by Step 3.
function step_case(; p_values, trial_F=Dict{Float64,Float64}(), B0=3.0,
                   epsilon=1.0, alpha=0.5, max_subproblem_iter=200, verbose=0, decrease_rule=:proximal_step)
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
        PDFPM_options(; verbose, epsilon, alpha, sigma=1.0, max_subproblem_iter, decrease_rule))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                               (iter, result) -> false; initial_B=[fill(B0, 1, 1)])
    return result, calls[]
end

@testset "PDFPM step tolerances: thresholds" begin
    # T = max_j tr(B_j) = max(1 + 2, 0.5 + 0.5) = 3, so 1 + T = 4.
    B = [Matrix(Diagonal([1.0, 2.0])), Matrix(Diagonal([0.5, 0.5]))]

    # ε = 3/4: ε/(3(1+T)) = 1/16 exactly, for both decrease rules.
    threshold, decrease = step_tolerances(2.0, B, 0.375; epsilon=0.75)
    @test threshold == 1 / 16
    @test decrease == 0.5 / 2 * 2.0 * 0.375^2  # (α/2)σ‖p-x‖², the default
    threshold, decrease = step_tolerances(2.0, B, 0.375; epsilon=0.75, decrease_rule=:epsilon_trace)
    @test threshold == 1 / 16
    @test decrease == 0.5 * 0.75^2 / (27 * 2.0 * 16)  # αε²/(27σ(1+T)²)

    # B = 0 gives T = 0: thresholds ε/3 and αε²/(27σ); ε = 3/4 gives 1/4.
    threshold, decrease = step_tolerances(1.0, [zeros(1, 1)]; epsilon=0.75, decrease_rule=:epsilon_trace)
    @test threshold == 0.25
    @test decrease == 0.5 * 0.75^2 / 27
    @test step_tolerances(1.0, [zeros(1, 1)], 0.5; epsilon=0.75) == (0.25, 0.5 / 2 * 0.25, 0.0)
end

@testset "PDFPM step tolerances: inner loop" begin
    # B0 = 3 and ε = 3/4: 1 + T = 4, stop threshold ε/(3·4) = 1/16 and decrease
    # αε²/(27σ·16) with σ = 1.
    epsilon = 0.75
    for verbose in (0, 1)
        # Step norm equal to the threshold proceeds to the decrease test and accepts;
        # the stationary candidate of the next iteration then satisfies Step 3.
        p = 2.0 - 1 / 16
        result, calls = step_case(; p_values=Dict(1 => p), epsilon, verbose)
        @test result.status === :proximal_step_tolerance
        @test result.x == [p] && result.iter == 2 && calls == 2

        # Step norm just below the threshold stops at the candidate.
        p = nextfloat(2.0 - 1 / 16)
        result, _ = step_case(; p_values=Dict(1 => p), epsilon, verbose)
        @test result.status === :proximal_step_tolerance
        @test result.x == [p] && result.iter == 1 && isnan(result.abs_theta)
        @test occursin("epsilon/(3(1+max_j tr(B_j)))", result.message)
    end

    # Decrease equal to the requirement is accepted; one ulp less is rejected, for
    # each rule. σ|p - x| = 3/2 passes the stop test; (α/2)σ‖p-x‖² = 9/16.
    p = 0.5
    for decrease_rule in (:proximal_step, :epsilon_trace)
        _, decrease = step_tolerances(1.0, [fill(3.0, 1, 1)], 1.5; epsilon, decrease_rule)
        decrease_rule === :proximal_step && @test decrease == 9 / 16
        for (F_p, accepted) in ((2.0 - decrease, true), (nextfloat(2.0 - decrease), false))
            result, _ = step_case(; p_values=Dict(1 => p), epsilon, decrease_rule,
                                  trial_F=Dict(p => F_p), max_subproblem_iter=1)
            @test result.status === (accepted ? :proximal_step_tolerance : :max_subproblem_iter)
            @test result.x == (accepted ? [p] : [2.0])
            accepted || @test result.required_decrease == decrease
        end
    end
end

@testset "PDFPM step tolerances: KNOW-008 counterexample" begin
    # f(x) = x²/2, h = 0, B = 0, x0 = ε/3, α = 1/2. The model with B = 0 is
    # solved in closed form: p = x - f'(x)/σ, θ = -f'(x)²/(2σ). The closed form
    # ignores B, which the BFGS update leaves nonfinite after the first step
    # (sᵀBs = 0); only k = 0, where B = 0 holds, is under test.
    for epsilon in (1.0, 1e-3)
        x0 = epsilon / 3
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
                          max_iter=1, max_subproblem_iter=50))
        MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                                   (iter, result) -> false; initial_B=[zeros(1, 1)])
        # σ|p - x| = ε/3 equals ε/(3(1+T)) and proceeds; F(p) = 0 ≤ ε²/18 - ε²/36,
        # with (α/2)σ‖p-x‖² = ε²/36, accepts at the first attempt (σ = 1).
        @test result.status === :max_iter
        @test result.iter == 1 && result.x == [0.0]
        @test Bs[1] == 0.0 && length(Bs) == 2
        @test result.n_f_evals == 2
    end
end
