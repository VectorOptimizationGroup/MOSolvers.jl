using MOSolvers
using LinearAlgebra
using Test

# Options decrease_rule and curvature of PDFPM.

@testset "PDFPM options decrease_rule and curvature" begin
    defaults = PDFPM_options()
    @test defaults.decrease_rule === :proximal_step && defaults.curvature === :bfgs
    params = MOSolvers._convert_pdfpm_options(Float64, PDFPM_options(decrease_rule=:epsilon_trace, curvature=:zero))
    @test params.decrease_rule === :epsilon_trace && params.curvature === :zero
    @test_throws ArgumentError PDFPM_options(decrease_rule=:armijo)
    @test_throws ArgumentError PDFPM_options(curvature=:identity)
    # The effective options, as a caller records them field by field.
    record = Dict(string(f) => getfield(defaults, f) for f in fieldnames(typeof(defaults)))
    @test record["decrease_rule"] === :proximal_step && record["curvature"] === :bfgs
    header = MOSolvers._solver_header_pairs(params)
    @test ("Decrease Rule" => "epsilon_trace") in header && ("Curvature" => "zero") in header
end

# f(x) = x², h = 0, x0 = 2, σ = 1, ε = 3/4. The subproblem is solved in closed form,
# p = x - f'(x)/(σ + B), and records the B it receives.
function curvature_case(curvature; max_iter=10)
    result = MOSolvers.OptimResult([2.0], [0.0])
    f!(F, x) = (F[1] = x[1]^2; F)
    Jf!(J, x; lambda=nothing) = (J[1, 1] = 2x[1]; J)
    h!(h, x) = (fill!(h, 0.0); (true, "h evaluated"))
    Bs = Float64[]
    function subproblem(x, J, h, method; sigma, B)
        push!(Bs, B[1][1, 1])
        g, c = J[1, 1], sigma + B[1][1, 1]
        return -g^2 / (2c), [x[1] - g / c], true, ""
    end
    options = MOSolvers._convert_pdfpm_options(Float64,
        PDFPM_options(; verbose=0, epsilon=0.75, alpha=0.5, sigma=1.0, max_iter, curvature))
    MOSolvers._PDFPM_optimize!(result, f!, Jf!, true, h!, 1, subproblem, options, (iter, result) -> false)
    return result, Bs
end

@testset "PDFPM curvature :zero" begin
    # B ≡ 0: p = -x is rejected at σ = 1 (F does not decrease), p = 0 accepted at σ = 2;
    # then the stationary candidate stops at Step 3 with threshold ε/3.
    result, Bs = curvature_case(:zero)
    @test result.status === :proximal_step_tolerance && result.x == [0.0]
    @test all(iszero, Bs) && length(Bs) == 3
    @test result.max_tr_B == 0.0 && result.step_threshold == 0.75 / 3
    # BFGS starts from the identity and updates after the accepted step.
    result, Bs = curvature_case(:bfgs)
    @test Bs[1] == 1.0 && any(!=(1.0), Bs[2:end])

    # Real backends: no trace of B other than zero, Step 3 at ε/3.
    evalf(x) = [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
    evalJf(x) = [2x[1] 2x[2]; 2(x[1] - 5.0) 2(x[2] - 5.0)]
    A = [[1.0 0.2; 0.1 1.0], [0.9 0.3; 0.2 1.1]]
    for delta in (0.0, 0.1)
        r = PDFPM(evalf, A, delta, [7.0, -3.0],
                  PDFPM_options(verbose=0, max_iter=50, epsilon=1e-4, store_level=:low, curvature=:zero);
                  evalJf, lb=fill(-10.0, 2), ub=fill(10.0, 2))
        @test r.iter >= 1 && all(iszero, r.max_tr_B_trace[2:end])
        @test r.max_tr_B == 0.0 && r.step_threshold == 1e-4 / 3
    end
end

@testset "Zero curvature matrices give the B = 0 subproblem" begin
    # Zero matrices, not B = nothing (the identity), represent B ≡ 0. The model of
    # PDFPM with B = 0 is the model of ProxGrad with ℓ = σ, built term by term the same.
    for (n, m, delta) in ((2, 2, 0.0), (3, 2, 0.05), (5, 3, 0.1)), sigma in (1.0, 4.0)
        A = [Matrix(1.0I, n, n) .+ (0.1 * j / n^2) .* reshape(1.0:n^2, n, n) for j in 1:m]
        x = collect(range(-1.0, 1.0; length=n))
        J = reshape(collect(range(-2.0, 3.0; length=m * n)), m, n)
        P = robust_subproblem_params(A, delta, fill(-10.0, n), fill(10.0, n))
        h = zeros(m)
        @test first(evalh!(h, x, P))
        zero_B = MOSolvers.subproblem_pdfpm(x, J, h, P; sigma, B=[zeros(n, n) for _ in 1:m], tol=1e-10)
        prox = MOSolvers.subproblem_proxgrad(x, J, h, P; sigma, tol=1e-10)
        identity_B = MOSolvers.subproblem_pdfpm(x, J, h, P; sigma, B=nothing, tol=1e-10)
        @test zero_B[3] && prox[3]
        @test isequal(zero_B[1], prox[1]) && isequal(zero_B[2], prox[2])
        @test zero_B[2] != identity_B[2]
    end
end
