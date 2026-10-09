using MOSolvers
using Test

@testset "Armijo nonfinite trials and limits ($T)" for T in (Float32, Float64)
    # Two objectives ensure that every component must be finite and satisfy Armijo.
    x, d, Fx, J = T[0], T[1], T[0, 0], fill(-one(T), 2, 1)
    for bad in T[NaN, Inf, -Inf], component in (1, 2)
        for threshold in (0.75, 0.3)
            trials = T[]
            function F(y)
                push!(trials, y[1])
                value = fill(-y[1], 2)
                y[1] > threshold && (value[component] = bad)
                return value, nothing
            end
            result = MOSolvers.armijo(F, x, d, Fx, J; factor=-one(T))
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
    trials = T[]
    function F_later_bad(y)
        push!(trials, y[1])
        value = length(trials) == 1 ? 1.0 : length(trials) == 2 ? NaN : -y[1]
        return fill(T(value), 2), nothing
    end
    result = MOSolvers.armijo(F_later_bad, x, d, Fx, J)
    @test result.success
    @test result.nfev == 3
    @test trials[3] == trials[2] / 2
    @test all(isfinite, result.F_final)

    for after_reduction in (false, true)
        F_error(y) = after_reduction && y[1] == 1.0 ?
                     (T[NaN, NaN], nothing) : (nothing, "injected evaluation error")
        result = MOSolvers.armijo(F_error, x, d, Fx, J)
        @test !result.success
        @test result.message == "injected evaluation error"
        @test result.nfev == (after_reduction ? 2 : 1)
    end

    # Repeated halving eventually gives fl(x + alpha*d) == x. Do not accept it
    # as sufficient decrease when the Armijo right-hand side also rounds to F(x).
    rounded_x = T[1]
    evaluations = Ref(0)
    function F_rounding(y)
        evaluations[] += 1
        return T[y == rounded_x ? 1 : NaN], nothing
    end
    options = MOSolvers.ArmijoParams(alpha_min=1e-30)
    result = MOSolvers.armijo(F_rounding, rounded_x, T[-1], T[1], ones(T, 1, 1); options)
    @test !result.success
    @test result.message == "The Armijo trial point is indistinguishable from x"
    @test result.nfev == evaluations[]
    @test 0 < result.nfev < options.max_iter
end

@testset "Armijo finite backtracking ($T)" for T in (Float32, Float64)
    x, d, Fx, J = T[0], T[3], T[1], fill(T(-2), 1, 1)
    initial = MOSolvers.LineSearchResult(F_final=Fx)
    @test initial.stp === zero(T)
    @test initial.F_final == Fx && initial.F_final !== Fx
    F(y) = ([(y[1] - one(T))^2], nothing)
    result = MOSolvers.armijo(F, x, d, Fx, J; factor=T(-6))
    @test result.success
    @test result.nfev == 2
    @test result.stp isa T
    @test eltype(result.F_final) == T
    @test result.stp ≈ one(T) / T(3)
    @test all(result.F_final .<= Fx .- T(1e-4) * result.stp * T(6))
    @test x == T[0] && d == T[3] && Fx == T[1]

    # Defaults and explicitly typed safeguards agree, including the fallback.
    for trial_value in (T(4), T(-5))
        step = MOSolvers.safeguarded_backtrack(one(T), trial_value, T(-6), one(T))
        explicit = MOSolvers.safeguarded_backtrack(one(T), trial_value, T(-6), one(T);
                                                  τ₁=T(0.05), τ₂=T(0.95))
        @test step isa T
        @test step == explicit
        @test zero(T) < step < one(T)
    end

    # Finite values that never satisfy Armijo also obey both limits.
    F_no_decrease(y) = (T[2], nothing)
    for (options, message, nfev) in (
        (MOSolvers.ArmijoParams(max_iter=0), "maximum number of iterations", 1),
        (MOSolvers.ArmijoParams(alpha_min=0.9), "did not converge", 1),
    )
        failed = MOSolvers.armijo(F_no_decrease, x, d, Fx, J; options)
        @test !failed.success
        @test occursin(message, failed.message)
        @test failed.nfev == nfev
    end
end
