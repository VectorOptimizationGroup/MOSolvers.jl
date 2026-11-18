using MOSolvers
using Test

include("regression_cases.jl")
include("regression_reference.jl")

# PDFPM with decrease_rule = :epsilon_trace and BFGS, ProxGrad and CondG reproduce
# commit 73e6de0 bit by bit (status, counts and the bits of x and F).
@testset "Regression against 73e6de0" begin
    common = (verbose=0, max_iter=30, subproblem_tol=1e-10)
    compared = 0
    for p in regression_problems(), delta in REGRESSION_DELTAS
        A = [regression_A(length(p.x0), j) for j in 1:length(p.evalf(p.x0))]
        runs = (PDFPM=() -> PDFPM(p.evalf, A, delta, p.x0,
                                  PDFPM_options(; common..., epsilon=1e-4, max_subproblem_iter=20,
                                                decrease_rule=:epsilon_trace, curvature=:bfgs);
                                  evalJf=p.evalJf, lb=p.lb, ub=p.ub),
                ProxGrad=() -> ProxGrad(p.evalf, p.evalJf, A, delta, p.x0, ProxGrad_options(; common...); lb=p.lb, ub=p.ub),
                CondG=() -> CondG(p.evalf, p.evalJf, A, delta, p.x0, CondG_options(; common...); lb=p.lb, ub=p.ub))
        for (solver, run) in pairs(runs)
            @test regression_record(run()) == REGRESSION_REFERENCE[(solver, p.name, delta)]
            compared += 1
        end
    end
    @test compared == length(REGRESSION_REFERENCE) == 30
    @test any(k -> k[1] === :PDFPM && REGRESSION_REFERENCE[k].n_sigma_doublings > 0, keys(REGRESSION_REFERENCE))
end
