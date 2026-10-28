using MOSolvers
using LinearAlgebra
using Test

# `store_level` chooses which traces are kept; it must not change the run.
# :low keeps the per-point measures and Fval, :medium adds x, and :high is
# currently the same as :medium.

const STORE_LEVEL_SOLVERS = (
    (CondG, CondG_options), (ProxGrad, ProxGrad_options), (PDFPM, PDFPM_options),
)

store_level_traces(r) = (; x=r.x_trace, Fval=r.Fval_trace, abs_theta=r.abs_theta_trace,
                         norm_progress=r.norm_progress_trace, sigma=r.sigma_trace,
                         max_tr_B=r.max_tr_B_trace)

function store_level_bk1(solver, opts, store_level)
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    options = opts(; verbose=0, max_iter=5, store_level)
    bounds = (; lb=fill(-500.0, 2), ub=fill(10000.0, 2))
    x0 = [-140.0, 6850.0]
    solver === PDFPM && return PDFPM(evalf, A, 0.0, x0, options; evalJf, bounds...)
    return solver(evalf, evalJf, A, 0.0, x0, options; bounds...)
end

@testset "store_level: option validation" begin
    for (_, opts) in STORE_LEVEL_SOLVERS
        @test opts().store_level === :none
        for level in MOSolvers.STORE_LEVELS
            @test opts(; store_level=level).store_level === level
        end
        @test_throws ArgumentError opts(; store_level=:full)
        @test_throws ArgumentError opts(; store_level=:None)
        # store_trace was removed without deprecation.
        @test_throws MethodError opts(; store_trace=true)
    end
    params = MOSolvers._convert_pdfpm_options(Float64, PDFPM_options(; store_level=:low))
    @test params.store_level === :low
    @test ("Store Level" => "low") in MOSolvers._solver_header_pairs(params)
end

@testset "store_level: trace sizes and alignment ($solver)" for (solver, opts) in STORE_LEVEL_SOLVERS
    runs = Dict(level => store_level_bk1(solver, opts, level) for level in MOSolvers.STORE_LEVELS)
    fields = (:x, :Fval, :x_init, :F_init, :iter, :abs_theta, :norm_progress,
              :n_f_evals, :n_Jf_evals, :status, :message)
    for level in MOSolvers.STORE_LEVELS
        @test all(isequal(getfield(runs[level], f), getfield(runs[:none], f)) for f in fields)
    end

    @test all(isempty, store_level_traces(runs[:none]))

    medium = store_level_traces(runs[:medium])
    k = runs[:medium].iter + 1
    @test runs[:medium].iter >= 1
    # ProxGrad records its initial x and Fval twice; the measures once.
    nx = solver === ProxGrad ? k + 1 : k
    nsigma = solver === PDFPM ? k : 0
    @test map(length, medium) == (; x=nx, Fval=nx, abs_theta=k, norm_progress=k,
                                  sigma=nsigma, max_tr_B=nsigma)
    @test medium.x[end] == runs[:medium].x && medium.Fval[end] == runs[:medium].Fval
    @test medium.x[1] == runs[:medium].x_init && medium.Fval[1] == runs[:medium].F_init
    solver === ProxGrad && @test medium.x[2] == medium.x[1] && medium.Fval[2] == medium.Fval[1]
    @test isnan(medium.norm_progress[1])
    solver === PDFPM && @test isnan(medium.sigma[1]) && isnan(medium.max_tr_B[1])

    # :low keeps the same entries as :medium, except x.
    low = store_level_traces(runs[:low])
    @test isempty(low.x)
    @test all(isequal(low[f], medium[f]) for f in (:Fval, :abs_theta, :norm_progress, :sigma, :max_tr_B))

    @test isequal(store_level_traces(runs[:high]), medium)

    # Stored arrays are copies, not aliases of the solver's buffers.
    @test medium.x[1] !== medium.x[end] && medium.Fval[1] !== medium.Fval[end]
end
