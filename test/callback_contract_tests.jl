using MOSolvers
using LinearAlgebra
using Test

# Contract of `callback(iter, result)`, on which external measurements rely.

const CB_A = [[1.0 0.2; 0.1 1.0], [0.9 0.3; 0.2 1.1]]
cb_f(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
cb_J(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
const CB_BOUNDS = (; lb=fill(-10.0, 2), ub=fill(10.0, 2))
const CB_X0 = [7.0, -3.0]

function cb_solve(method, delta, callback; max_iter=8, opt_tol=0.0)
    kw = (; verbose=0, max_iter, store_level=:medium)
    method === :pdfpm && return PDFPM(cb_f, CB_A, delta, CB_X0, PDFPM_options(; kw..., epsilon=1e-4);
                                      evalJf=cb_J, CB_BOUNDS..., callback)
    solver, opts = method === :proxgrad ? (ProxGrad, ProxGrad_options) : (CondG, CondG_options)
    return solver(cb_f, cb_J, CB_A, delta, CB_X0, opts(; kw..., opt_tol); CB_BOUNDS..., callback)
end

# Trace index of iteration k: ProxGrad records its initial x and Fval twice.
cb_index(method, k) = method === :proxgrad ? k + 2 : k + 1

@testset "Callback sees each accepted iterate ($method, delta=$delta)" for method in (:condg, :proxgrad, :pdfpm), delta in (0.0, 0.1)
    seen = NamedTuple[]
    r = cb_solve(method, delta, (k, res) -> (push!(seen, (; k, x=copy(res.x), F=copy(res.Fval), it=res.iter, status=res.status)); nothing))
    @test r.iter >= 1
    # A Step 3 exit and a θ = 0 exit come after the last callback; the others do not.
    last_visited = r.status === :proximal_step_tolerance ? r.iter - 1 : r.iter
    @test [s.k for s in seen] == collect(1:last_visited)
    for s in seen
        @test s.x == r.x_trace[cb_index(method, s.k)]
        @test s.F == r.Fval_trace[cb_index(method, s.k)]
        @test s.it == s.k && s.status === :running
    end
    if r.status === :proximal_step_tolerance
        @test isempty(seen) || seen[end].x != r.x
    elseif !isempty(seen)
        @test seen[end].x == r.x
    end

    # Abort at the first iteration that reaches it returns that iterate.
    if !isempty(seen)
        calls = Ref(0)
        aborted = cb_solve(method, delta, (k, res) -> (calls[] += 1; k == seen[1].k))
        @test aborted.status === :callback && !converged(aborted)
        @test calls[] == 1 && aborted.iter == seen[1].k
        @test aborted.x == seen[1].x && aborted.Fval == seen[1].F
    end
end

@testset "Only exactly true aborts" begin
    for method in (:condg, :proxgrad, :pdfpm), value in (1, "true", missing, false, nothing)
        r = cb_solve(method, 0.1, (k, res) -> value; max_iter=1)
        @test r.status !== :callback
    end
end

@testset "CondG and ProxGrad: a satisfied criterion precedes the callback" begin
    # With opt_tol > 0, the iteration that satisfies it is accepted but not visited.
    for method in (:condg, :proxgrad)
        seen = Int[]
        r = cb_solve(method, 0.1, (k, res) -> (push!(seen, k); nothing); max_iter=500, opt_tol=1e-2)
        @test r.status === :optimality_tolerance
        @test seen == collect(1:(r.iter - 1))
    end
end

# PDFPM with controlled candidates: x0 = 2, F(x) = x²/2, ε = 3/4; call c of the
# subproblem returns p_values[c], or x. `fails` is the set of candidates with no decrease.
function cb_pdfpm_case(; p_values, callback, max_iter=5, max_subproblem_iter=3)
    result = MOSolvers.OptimResult([2.0], [0.0])
    f!(F, x) = (F[1] = x[1] > 2.0 ? 100.0 : x[1]^2 / 2; F)
    Jf!(J, x; lambda=nothing) = (J[1, 1] = x[1]; J)
    h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
    calls = Ref(0)
    function subproblem(x, J, h, method; sigma, B)
        calls[] += 1
        return -1.0, [get(p_values, calls[], x[1])], true, ""
    end
    options = MOSolvers._convert_pdfpm_options(Float64,
        PDFPM_options(; verbose=0, epsilon=0.75, alpha=0.5, sigma=1.0, max_iter, max_subproblem_iter))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options, callback;
                               initial_B=[fill(3.0, 1, 1)])
    return result, calls[]
end

@testset "PDFPM: callback comes before the doublings of the next iteration" begin
    # Call 1 is accepted (iteration 1). Calls 2 and 3 are rejected and call 4 accepted (iteration 2).
    seen = NamedTuple[]
    cb(k, res) = (push!(seen, (; k, x=res.x[1], doublings=res.n_sigma_doublings)); nothing)
    result, _ = cb_pdfpm_case(; p_values=Dict(1 => 1.0, 2 => 3.0, 3 => 3.0, 4 => 0.5), callback=cb, max_iter=2)
    @test result.status === :max_iter
    @test seen[1] == (; k=1, x=1.0, doublings=0)
    @test seen[2] == (; k=2, x=0.5, doublings=2)
end

@testset "PDFPM: max_subproblem_iter returns the last accepted iterate" begin
    seen = Float64[]
    cb(k, res) = (push!(seen, res.x[1]); nothing)
    result, _ = cb_pdfpm_case(; p_values=Dict(1 => 1.0, (c => 3.0 for c in 2:9)...), callback=cb)
    @test result.status === :max_subproblem_iter && !converged(result)
    @test seen == [1.0] && result.x == [1.0] && result.Fval == [0.5] && result.iter == 1
    @test result.n_sigma_doublings == 2

    # Without any accepted iteration, that is x0, and no callback ran.
    empty!(seen)
    result, _ = cb_pdfpm_case(; p_values=Dict(c => 3.0 for c in 1:9), callback=cb)
    @test result.status === :max_subproblem_iter
    @test isempty(seen) && result.x == [2.0] && result.iter == 0

    # Abort takes effect before the rejected candidates of the next iteration.
    result, calls = cb_pdfpm_case(; p_values=Dict(1 => 1.0, (c => 3.0 for c in 2:9)...), callback=(k, res) -> true)
    @test result.status === :callback && result.x == [1.0] && calls == 2 && result.n_sigma_doublings == 0
end
