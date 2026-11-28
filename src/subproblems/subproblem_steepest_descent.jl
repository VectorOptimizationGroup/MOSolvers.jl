# Subproblems of the multiobjective steepest descent method (Fliege and Svaiter, 2000).
#
# With the Jacobian JF (m × n, row i = ∇f_i(x)'), the steepest descent direction is
# d = -JF'γ, where γ minimizes ½‖JF'γ‖² over the unit simplex (the dual of problem (2)
# of Fliege and Svaiter). The optimal value of (2) is θ = -½‖d‖².

"""
    simplex_qp(M::AbstractMatrix{T}) -> (γ, success, message)

Minimize `½ γ'Mγ` over the unit simplex for a symmetric positive semidefinite `M`
of order `m`. Exact for `m ≤ 6` (closed form for `m = 2`, enumeration of the faces
otherwise); HiGHS for `m ≥ 7`.
"""
function simplex_qp(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    m = size(M, 1)
    size(M, 2) == m || throw(DimensionMismatch("M must be square"))
    m == 1 && return [one(T)], true, ""
    m == 2 && return _simplex_qp_two(M), true, ""
    m <= 6 && return _simplex_qp_faces(M), true, ""
    return _simplex_qp_highs(M)
end

# Minimizer of q(t) = ½‖t g₁ + (1 - t) g₂‖² on [0, 1], with M = Gram matrix of g₁, g₂.
function _simplex_qp_two(M::AbstractMatrix{T}) where {T}
    a = M[1, 1] - 2 * M[1, 2] + M[2, 2]  # ‖g₁ - g₂‖² ≥ 0
    t = if a > zero(T)
        clamp((M[2, 2] - M[1, 2]) / a, zero(T), one(T))
    else
        # g₁ = g₂ up to rounding: q is constant; take the endpoint with the smaller value
        M[1, 1] <= M[2, 2] ? one(T) : zero(T)
    end
    return T[t, one(T) - t]
end

# Every minimizer of a convex quadratic over the simplex solves the KKT system on the face
# spanned by its support: [M_SS 1; 1' 0] [γ_S; -μ] = [0; 1] with γ_S ≥ 0. Solving that
# system on each of the 2^m - 1 faces and keeping the feasible solution with the least value
# gives a global minimizer.
function _simplex_qp_faces(M::AbstractMatrix{T}) where {T}
    m = size(M, 1)
    best_value = T(Inf)
    best = zeros(T, m)
    for mask in 1:(2^m - 1)
        S = [i for i in 1:m if (mask >> (i - 1)) & 1 == 1]
        k = length(S)
        K = zeros(T, k + 1, k + 1)
        K[1:k, 1:k] .= view(M, S, S)
        K[1:k, k + 1] .= one(T)
        K[k + 1, 1:k] .= one(T)
        rhs = zeros(T, k + 1)
        rhs[k + 1] = one(T)
        sol = try
            K \ rhs
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
            pinv(K) * rhs
        end
        all(isfinite, sol) || continue
        norm(K * sol - rhs) <= sqrt(eps(T)) * (1 + norm(K)) || continue
        γS = view(sol, 1:k)
        minimum(γS) >= -sqrt(eps(T)) || continue
        γ = zeros(T, m)
        γ[S] .= max.(γS, zero(T))
        γ ./= sum(γ)
        value = dot(γ, M * γ) / 2
        if value < best_value
            best_value = value
            best = γ
        end
    end
    return best
end

function _simplex_qp_highs(M::AbstractMatrix{T}) where {T}
    m = size(M, 1)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, γ[1:m] >= 0)
    @constraint(model, sum(γ) == 1)
    Mf = Float64.((M .+ M') ./ 2)
    @objective(model, Min, 0.5 * sum(Mf[i, l] * γ[i] * γ[l] for i in 1:m, l in 1:m))
    optimize!(model)
    if !is_solved_and_feasible(model)
        return fill(T(NaN), m), false, "HiGHS: $(termination_status(model))"
    end
    γv = max.(T.(value.(γ)), zero(T))
    return γv ./ sum(γv), true, ""
end

"""
    steepest_descent_direction!(d, JF) -> (γ, success, message)

Write the steepest descent direction `d = -JF'γ` and return the weights `γ`.
`abs(θ) = ½‖d‖²` is the optimal value of problem (2) of Fliege and Svaiter in absolute value.
"""
function steepest_descent_direction!(d::AbstractVector{T}, JF::AbstractMatrix{T}) where {T<:AbstractFloat}
    γ, success, message = simplex_qp(Symmetric(JF * JF'))
    success || return γ, false, message
    mul!(d, JF', γ)
    d .*= -one(T)
    return γ, true, ""
end

"""
    max_norm_measure(JF) -> (value, success, message)

`min ‖JF'γ‖_∞` over the unit simplex, by a linear program (HiGHS); for one objective,
`‖∇f(x)‖_∞`. It is zero exactly at Pareto critical points and equals `-β(x)`, the optimal
value of problem (5) of Fliege and Svaiter with the constraint `‖v‖₁ ≤ 1`.
"""
function max_norm_measure(JF::AbstractMatrix{T}) where {T<:AbstractFloat}
    m, n = size(JF)
    m == 1 && return norm(view(JF, 1, :), Inf), true, ""
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, γ[1:m] >= 0)
    @variable(model, t)
    @constraint(model, sum(γ) == 1)
    J = Float64.(JF)
    for j in 1:n
        e = sum(J[i, j] * γ[i] for i in 1:m)
        @constraint(model, e <= t)
        @constraint(model, -e <= t)
    end
    @objective(model, Min, t)
    optimize!(model)
    if !is_solved_and_feasible(model)
        return T(NaN), false, "HiGHS: $(termination_status(model))"
    end
    γv = max.(T.(value.(γ)), zero(T))
    γv ./= sum(γv)
    return norm(JF' * γv, Inf), true, ""
end
