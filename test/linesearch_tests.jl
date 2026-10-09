using MOSolvers
using Test

@testset "Armijo nonfinite trials and limits" begin
    # Two objectives ensure that every component must be finite and satisfy Armijo.
    x, d, Fx, J = [0.0], [1.0], [0.0, 0.0], fill(-1.0, 2, 1)
    for bad in (NaN, Inf, -Inf), component in (1, 2)
        for threshold in (0.75, 0.3)
            trials = Float64[]
            function F(y)
                push!(trials, y[1])
                value = fill(-y[1], 2)
                y[1] > threshold && (value[component] = bad)
                return value, nothing
            end
            result = MOSolvers.armijo(F, x, d, Fx, J; factor=-1.0)
            @test result.success
            @test 0 < result.stp <= threshold
            @test all(isfinite, result.F_final)
            @test all(result.F_final .<= Fx .- 1e-4 * result.stp)
            @test result.nfev == length(trials)
            @test trials == (threshold == 0.75 ? [1.0, 0.5] : [1.0, 0.5, 0.25])
            @test x == [0.0] && Fx == [0.0, 0.0]
        end

        F_bad(y) = (value = fill(-y[1], 2); value[component] = bad; (value, nothing))
        for limit in (0, 2)
            options = MOSolvers.ArmijoParams(max_iter=limit)
            result = MOSolvers.armijo(F_bad, x, d, Fx, J; options)
            @test !result.success
            @test occursin("maximum number of iterations", result.message)
            @test result.nfev == limit + 1
        end
        options = MOSolvers.ArmijoParams(alpha_min=0.25)
        result = MOSolvers.armijo(F_bad, x, d, Fx, J; options)
        @test !result.success
        @test result.message == "The Armijo line search did not converge"
        @test result.nfev == 2
    end

    # The first finite trial fails Armijo; the next is nonfinite; a later one recovers.
    trials = Float64[]
    function F_later_bad(y)
        push!(trials, y[1])
        value = length(trials) == 1 ? 1.0 : length(trials) == 2 ? NaN : -y[1]
        return fill(value, 2), nothing
    end
    result = MOSolvers.armijo(F_later_bad, x, d, Fx, J)
    @test result.success
    @test result.nfev == 3
    @test trials[3] == trials[2] / 2
    @test all(isfinite, result.F_final)

    for after_reduction in (false, true)
        F_error(y) = after_reduction && y[1] == 1.0 ?
                     ([NaN, NaN], nothing) : (nothing, "injected evaluation error")
        result = MOSolvers.armijo(F_error, x, d, Fx, J)
        @test !result.success
        @test result.message == "injected evaluation error"
        @test result.nfev == (after_reduction ? 2 : 1)
    end

    # Repeated halving eventually gives fl(x + alpha*d) == x. Do not accept it
    # as sufficient decrease when the Armijo right-hand side also rounds to F(x).
    rounded_x = [1.0]
    evaluations = Ref(0)
    function F_rounding(y)
        evaluations[] += 1
        return [y == rounded_x ? 1.0 : NaN], nothing
    end
    options = MOSolvers.ArmijoParams(alpha_min=1e-30)
    result = MOSolvers.armijo(F_rounding, rounded_x, [-1.0], [1.0], ones(1, 1); options)
    @test !result.success
    @test result.message == "The Armijo trial point is indistinguishable from x"
    @test result.nfev == evaluations[]
    @test 0 < result.nfev < options.max_iter
end
