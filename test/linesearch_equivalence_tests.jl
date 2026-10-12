using MOSolvers
using MultiObjectiveLineSearches
using LinearAlgebra
using Test

# MOSolvers.armijo and BackTracking from MultiObjectiveLineSearches must make the
# same trials and report the same outcome. On failure only the representation
# differs: armijo returns a sentinel step, BackTracking the last candidate.
const ARMIJO_REASONS = Dict(
    "A step size that satisfies the Armijo condition was found" => :success,
    "The maximum number of iterations for the Armijo line search was reached" => :max_iterations,
    "The Armijo line search did not converge" => :minimum_step,
    "The Armijo trial point is indistinguishable from x" => :indistinguishable_point)

# `f(y)` returns the objective vector, or an error message.
function compare_with_backtracking(f, x, d, Fx, J, factor; options...)
    T = eltype(x)
    slopes = [dot(view(J, i, :), d) for i in axes(J, 1)]  # as computed by armijo
    native, external = Vector{T}[], Vector{T}[]
    function F(y)
        push!(native, copy(y))
        v = f(y)
        return v isa String ? (nothing, v) : (v, nothing)
    end
    function evaluate!(Fy, y)
        push!(external, copy(y))
        v = f(y)
        v isa String && return v
        Fy .= v
        return nothing
    end
    params = MOSolvers.ArmijoParams(; options...)
    old = MOSolvers.armijo(F, x, d, Fx, J; factor, options=params)
    spec = BackTracking(; initial_step=InitialStatic(alpha=params.alpha_init), c_1=params.rho,
                        max_iters=params.max_iter, minstep=params.alpha_min,
                        maxstep=params.alpha_max)
    y, Fy = similar(x), similar(Fx)
    new = perform_linesearch!(configure_linesearch(spec, T), evaluate!, x, d, Fx, slopes,
                              y, Fy; factor)
    @test isequal(native, external)
    @test get(ARMIJO_REASONS, old.message, :evaluation_failure) === new.reason
    @test old.success == new.success && old.nfev == new.nfev
    if new.success
        @test old.stp === new.alpha
        @test isequal(old.F_final, Fy) && isequal(y, native[end])
    elseif new.reason === :evaluation_failure
        @test old.message == new.message
    end
    return new
end

@testset "armijo and BackTracking agree ($T)" for T in (Float32, Float64)
    sq(y) = [(y[1] - 1)^2]
    two(y) = [(y[1] - 1)^2, (y[1] - 2)^2]
    col(v...) = reshape(T[v...], :, 1)
    # factor = -15 makes the association of c_1*alpha*factor change the last bit
    # in both precisions; armijo computes (c_1*alpha)*factor.
    thr = T(1e-4) * T(0.3) * T(-15)
    cases = [
        # objective, x, d, Fx, Jacobian, factor, options, expected reason
        (sq, 0, 3, [1], col(-2), -6, (;), :success),              # interpolation kept
        (sq, 0, 40, [1], col(-2), -80, (;), :success),            # below tau_1*alpha: halve
        (two, 0, 6, [1, 4], col(-2, -4), -3, (;), :success),      # first violator, factor apart
        (y -> 15 < y[1] < 30 ? [T(Inf)] : sq(y), 0, 40, [1], col(-2), -80, (;), :success),
        (y -> [sq(y)[1], y[1] > 2.5 ? T(NaN) : (y[1] - 2)^2], 0, 3, [1, 4], col(-2, -4), -6,
         (;), :success),                                          # nonfinite elsewhere halves
        (sq, 0, 40, [1], col(-2), -80, (; alpha_min=0.2), :minimum_step),
        (y -> [y[1]], 0, 1, [0], col(1), -1, (; alpha_min=0.125), :minimum_step),
        (y -> [-y[1]], 0, 1, [0], col(-1), -1e5, (; max_iter=3), :max_iterations),
        (y -> [y[1]], 1, 4eps(T), [1], col(1), -1, (;), :indistinguishable_point),
        (y -> y[1] < 30 ? "reported error" : sq(y), 0, 40, [1], col(-2), -80, (;),
         :evaluation_failure),
        (y -> [thr], 0, 1, [0], col(-1), -15, (; alpha_init=0.3), :success),
        (y -> [nextfloat(thr)], 0, 1, [0], col(-1), -15, (; alpha_init=0.3), :success),
    ]
    for (f, x, d, Fx, J, factor, options, reason) in cases
        result = compare_with_backtracking(f, T[x], T[d], T.(Fx), J, T(factor); options...)
        @test result.reason === reason
    end
end
