using MOSolvers
using LinearAlgebra
using Test

# Diagnostics of PDFPM's last step test and the option subproblem_tol.

const STEP_FIELDS = (:sigma, :max_tr_B, :step_measure, :step_threshold, :required_decrease, :actual_decrease)

# One-dimensional run with h = 0, x0 = 2, F(x) = x²/2 unless `trial_F` says otherwise,
# B0 = 3, ε = 3/4, α = 1/2 and σ = 1. Call c of the subproblem returns `p_values[c]`, or x.
function diagnostics_case(; p_values, trial_F=Dict{Float64,Float64}(), max_iter=1, max_subproblem_iter=200)
    result = MOSolvers.OptimResult([2.0], [0.0])
    f!(F, x) = (F[1] = get(trial_F, x[1], x[1]^2 / 2); F)
    Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1]; J)
    h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
    calls = Ref(0)
    function subproblem(x, J, h, method; sigma, B)
        calls[] += 1
        return -1.0, [get(p_values, calls[], x[1])], true, ""
    end
    options = MOSolvers._convert_pdfpm_options(Float64,
        PDFPM_options(; verbose=0, epsilon=0.75, alpha=0.5, sigma=1.0, max_iter, max_subproblem_iter))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options,
                               (iter, result) -> false; initial_B=[fill(3.0, 1, 1)])
    return result
end

# Default decrease of Step 4, (α/2)σ‖p-x‖², with α = 1/2.
required(sigma, step) = 0.5 / 2 * sigma * step^2

@testset "PDFPM diagnostics of the last step test" begin
    # No test yet.
    result = diagnostics_case(; p_values=Dict(1 => 1.0), max_iter=0)
    @test all(f -> isnan(getfield(result, f)), STEP_FIELDS)
    @test result.n_sigma_doublings == 0

    # p = 3 is rejected (F grows), then p = 1 is accepted with σ = 2.
    result = diagnostics_case(; p_values=Dict(1 => 3.0, 2 => 1.0))
    @test result.status === :max_iter && result.x == [1.0]
    @test result.sigma == 2.0 && result.max_tr_B == 3.0
    @test result.step_measure == 2.0 && result.step_threshold == 1 / 16
    @test result.required_decrease == required(2.0, 1.0)
    @test result.actual_decrease == 1.5
    @test result.n_sigma_doublings == 1

    # Exit at Step 3: no decrease was tested in that pass.
    result = diagnostics_case(; p_values=Dict(1 => 2.0 - 1 / 64))
    @test result.status === :proximal_step_tolerance
    @test result.step_measure == 1 / 64 && result.step_threshold == 1 / 16
    @test isnan(result.actual_decrease) && result.n_sigma_doublings == 0

    # Exhausted inner loop: last rejected test, with each doubling counted.
    result = diagnostics_case(; p_values=Dict(c => 3.0 for c in 1:4), max_subproblem_iter=4)
    @test result.status === :max_subproblem_iter
    @test result.sigma == 8.0 && result.n_sigma_doublings == 3
    @test result.step_measure == 8.0
    @test result.actual_decrease == 2.0 - 4.5
    @test result.required_decrease == required(8.0, 1.0)

    # A nonfinite trial gives a nonfinite measure, and it is rejected.
    result = diagnostics_case(; p_values=Dict(1 => 1.0, 2 => 1.5), trial_F=Dict(1.0 => NaN))
    @test result.x == [1.5] && result.n_sigma_doublings == 1 && result.actual_decrease == 2.0 - 1.125
    result = diagnostics_case(; p_values=Dict(1 => 1.0), trial_F=Dict(1.0 => NaN), max_subproblem_iter=1)
    @test result.status === :max_subproblem_iter && isnan(result.actual_decrease)
end

@testset "Diagnostics with real backends" begin
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [[1.0 0.2; 0.1 1.0], [0.9 0.3; 0.2 1.1]]
    bounds = (; lb=fill(-10.0, 2), ub=fill(10.0, 2))
    x0 = [7.0, -3.0]
    fields = (:x, :Fval, :iter, :abs_theta, :norm_progress, :n_f_evals, :n_Jf_evals, :status, :message,
              STEP_FIELDS..., :n_sigma_doublings)
    for delta in (0.0, 0.1)
        runs = map((:none, :medium)) do store_level
            PDFPM(evalf, A, delta, x0, PDFPM_options(; verbose=0, max_iter=50, epsilon=1e-4, store_level); evalJf, bounds...)
        end
        @test all(isequal(getfield(runs[1], f), getfield(runs[2], f)) for f in fields)
        r = runs[1]
        @test r.iter >= 1 && isfinite(r.step_measure) && isfinite(r.step_threshold)
        @test r.sigma == 2.0^r.n_sigma_doublings
        @test r.step_threshold == 1e-4 / (3 * (1 + r.max_tr_B))
        # σ is a power of 2, so step_measure/σ recovers ‖p-x‖ exactly.
        @test r.required_decrease == 0.1 / 2 * r.sigma * (r.step_measure / r.sigma)^2
        @test r.max_tr_B == runs[2].max_tr_B_trace[end]
    end
    for (solver, opts) in ((CondG, CondG_options), (ProxGrad, ProxGrad_options))
        r = solver(evalf, evalJf, A, 0.1, x0, opts(verbose=0, max_iter=5); bounds...)
        @test all(f -> isnan(getfield(r, f)), STEP_FIELDS) && r.n_sigma_doublings == 0
    end
end

@testset "Option subproblem_tol" begin
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [[1.0 0.2; 0.1 1.0], [0.9 0.3; 0.2 1.1]]
    bounds = (; lb=fill(-10.0, 2), ub=fill(10.0, 2))
    x0 = [7.0, -3.0]
    cases = ((CondG_options, MOSolvers._convert_condg_options, o -> CondG(evalf, evalJf, A, 0.1, x0, o; bounds...)),
             (ProxGrad_options, MOSolvers._convert_proxgrad_options, o -> ProxGrad(evalf, evalJf, A, 0.1, x0, o; bounds...)),
             (PDFPM_options, MOSolvers._convert_pdfpm_options, o -> PDFPM(evalf, A, 0.1, x0, o; evalJf, bounds...)))
    for (opts, convert_options, solve) in cases
        @test convert_options(Float64, opts()).subproblem_tol == sqrt(eps(Float64))
        @test convert_options(Float32, opts()).subproblem_tol == sqrt(eps(Float32))
        @test convert_options(Float64, opts(subproblem_tol=1e-10)).subproblem_tol == 1e-10
        @test ("Subproblem Tol" => "1.00e-10") in MOSolvers._global_header_pairs(convert_options(Float64, opts(subproblem_tol=1e-10)), 2, 2)
        for bad in (0.0, -1e-8, Inf, NaN)
            @test_throws ArgumentError opts(subproblem_tol=bad)
        end
        fields = (:x, :Fval, :iter, :status)
        kw = (; verbose=0, max_iter=8, store_level=:medium)
        # The default is the former fixed value, bit by bit.
        default, explicit = solve(opts(; kw...)), solve(opts(; kw..., subproblem_tol=sqrt(eps(Float64))))
        @test all(isequal(getfield(default, f), getfield(explicit, f)) for f in fields)
        @test default.x_trace == explicit.x_trace
        # The value reaches the solver of the subproblem.
        tight = solve(opts(; kw..., subproblem_tol=1e-10))
        @test tight.status ∉ (:subproblem_failure, :evaluation_failure, :numerical_failure)
        # Below the smallest tolerance HiGHS accepts, CondG uses that smallest one.
        @test solve(opts(; kw..., subproblem_tol=1e-12)).status ∉ (:subproblem_failure, :evaluation_failure, :numerical_failure)
        loose = solve(opts(; kw..., subproblem_tol=1e-3))
        opts === CondG_options || @test tight.x_trace != loose.x_trace
    end
end
