using MOSolvers
using LinearAlgebra
using Test

# Traces of PDFPM's σ and maxⱼ tr(Bⱼ). Entry k of `sigma_trace` and
# `max_tr_B_trace` belongs to entry k of the other traces.

# One-dimensional run with h = 0, x0 = 2, F(x) = x²/2 unless `trial_F` says
# otherwise, and B0 = 3. Call c of the subproblem returns `p_values[c]` with
# θ = -1, or x with θ = 0 (optimality) when c is not a key.
function trace_case(rule; p_values, trial_F=Dict{Float64,Float64}(),
                    max_subproblem_iter=200, store_trace=true)
    result = MOSolvers.OptimResult([2.0], [0.0])
    f!(F, x) = (F[1] = get(trial_F, x[1], x[1]^2 / 2); F)
    Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1]; J)
    h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
    Bs = Float64[]
    function subproblem(x, J, h, method; sigma, B)
        push!(Bs, B[1][1, 1])
        haskey(p_values, length(Bs)) || return 0.0, [x[1]], true, ""
        return -1.0, [p_values[length(Bs)]], true, ""
    end
    options = MOSolvers._convert_pdfpm_options(Float64,
        PDFPM_options(; verbose=0, epsilon=1.0, alpha=0.5, sigma=1.0, opt_tol=1e-8,
                      max_subproblem_iter, store_trace, step_tolerance_rule=rule))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                               (iter, result) -> false; initial_B=[fill(3.0, 1, 1)])
    return result, Bs
end

trace_lengths(r) = map(length, (r.x_trace, r.Fval_trace, r.abs_theta_trace,
                                r.norm_progress_trace, r.sigma_trace, r.max_tr_B_trace))

@testset "PDFPM σ and tr(B) traces: controlled cases" begin
    @test isempty(MOSolvers.OptimResult([1.0], [0.0]).sigma_trace)
    @test isempty(MOSolvers.OptimResult([1.0], [0.0]).max_tr_B_trace)
    for rule in (:epsilon, :trace_scaled)
        # p = 0.5 gives no decrease (F = 2 = F(x0)) and is rejected: σ = 2.
        # p = 1 passes both rules' stop and decrease tests with σ = 2 and T = 3;
        # the BFGS update then changes B before the next subproblem.
        result, Bs = trace_case(rule; p_values=Dict(1 => 0.5, 2 => 1.0), trial_F=Dict(0.5 => 2.0))
        @test result.status === :optimality_tolerance && result.iter == 1
        @test trace_lengths(result) == ntuple(_ -> 2, 6)
        @test isequal(result.sigma_trace, [NaN, 2.0])
        @test isequal(result.max_tr_B_trace, [NaN, 3.0])
        @test Bs[1:2] == [3.0, 3.0] && Bs[3] != 3.0

        # Proximal-step exit after one rejection: σ‖p-x‖ = 2·2⁻⁶ stops under
        # both rules; the entry holds the σ and T of that stop test.
        result, _ = trace_case(rule; p_values=Dict(1 => 0.5, 2 => 2.0 - 2.0^-6), trial_F=Dict(0.5 => 2.0))
        @test result.status === :proximal_step_tolerance && result.iter == 1
        @test trace_lengths(result) == ntuple(_ -> 2, 6)
        @test isequal(result.abs_theta_trace, [1.0, NaN])
        @test isequal(result.sigma_trace, [NaN, 2.0])
        @test isequal(result.max_tr_B_trace, [NaN, 3.0])

        # Exhausted inner loop: no point is recorded after x0.
        result, _ = trace_case(rule; p_values=Dict(1 => 0.5), trial_F=Dict(0.5 => 2.0),
                               max_subproblem_iter=1)
        @test result.status === :max_subproblem_iter
        @test trace_lengths(result) == ntuple(_ -> 1, 6)
        @test isequal(result.sigma_trace, [NaN]) && isequal(result.max_tr_B_trace, [NaN])

        result, _ = trace_case(rule; p_values=Dict(1 => 0.5, 2 => 1.0), trial_F=Dict(0.5 => 2.0),
                               store_trace=false)
        @test trace_lengths(result) == ntuple(_ -> 0, 6)
    end
end

@testset "PDFPM σ and tr(B) traces: BK1" begin
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    bounds = (; lb=fill(-500.0, 2), ub=fill(10000.0, 2))
    fields = (:x, :Fval, :iter, :abs_theta, :norm_progress, :n_f_evals, :n_Jf_evals, :status, :message)
    for rule in (:epsilon, :trace_scaled), jac in ((; evalJf), (;))
        runs = map((true, false)) do store_trace
            options = PDFPM_options(; verbose=0, max_iter=50, opt_tol=1e-6, store_trace,
                                    step_tolerance_rule=rule)
            PDFPM(evalf, A, 0.0, [-140.0, 6850.0], options; jac..., bounds...)
        end
        traced, plain = runs
        @test traced.iter >= 1
        @test trace_lengths(traced) == ntuple(_ -> traced.iter + 1, 6)
        @test isnan(traced.sigma_trace[1]) && isnan(traced.max_tr_B_trace[1])
        # The first step uses B⁰ = I in R²: T = 2.
        @test traced.max_tr_B_trace[2] == 2.0
        @test all(isfinite, traced.max_tr_B_trace[2:end])
        # σ starts at 1 and is only doubled.
        @test issorted(traced.sigma_trace[2:end]) && traced.sigma_trace[2] >= 1.0
        @test all(s -> ispow2(s), traced.sigma_trace[2:end])
        @test trace_lengths(plain) == ntuple(_ -> 0, 6)
        @test all(isequal(getfield(traced, f), getfield(plain, f)) for f in fields)
    end
end

@testset "PDFPM σ and tr(B) traces: empty for CondG and ProxGrad" begin
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    bounds = (; lb=fill(-500.0, 2), ub=fill(10000.0, 2))
    for (solver, opts) in ((CondG, CondG_options), (ProxGrad, ProxGrad_options))
        result = solver(evalf, evalJf, A, 0.0, [-140.0, 6850.0],
                        opts(verbose=0, max_iter=5, store_trace=true); bounds...)
        @test !isempty(result.x_trace) && !isempty(result.abs_theta_trace)
        @test isempty(result.sigma_trace) && isempty(result.max_tr_B_trace)
    end
end
