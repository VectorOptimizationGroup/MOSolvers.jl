# Values of the `store_level` option, from no trace to the most detailed one.
const STORE_LEVELS = (:none, :low, :medium, :high)

"""
    mutable struct OptimResult{T<:AbstractFloat}

Container structure holding the results produced by an optimization solver.

# Fields
- `x::Vector{T}`: Final iterate returned by the solver.
- `Fval::Vector{T}`: Value of the composite objective `F(x)` at `x`.
- `x_init::Vector{T}`: Initial point provided to the solver.
- `F_init::Vector{T}`: Value of the composite objective at `x_init`.
- `iter::Int`: Number of iterations performed.
- `abs_theta::T`: Absolute value of the model measure θ at the returned iterate;
  `NaN` when it has not been computed there (PDFPM's proximal-step exit).
- `norm_progress::T`: Normalised ∞-norm progress between consecutive iterates.
- `n_f_evals::Int`: Number of objective evaluations (`f!`).
- `n_Jf_evals::Int`: Number of Jacobian evaluations (`Jf!`).
- `total_time::Float64`: Total wall-clock time spent by the solver (seconds).
- `subproblem_time::Float64`: Cumulative time spent solving sub-problems (seconds).
- `x_trace::Vector{Vector{T}}`: History of iterates (`store_level` `:medium` or `:high`).
- `Fval_trace::Vector{Vector{T}}`: History of `Fval` values (`store_level` other than `:none`).
- `abs_theta_trace::Vector{T}`: History of `abs_theta` values (`store_level` other than `:none`).
- `norm_progress_trace::Vector{T}`: History of `norm_progress` values (`store_level` other than `:none`).
- `sigma_trace::Vector{T}`: PDFPM only, `store_level` other than `:none`: per recorded point, the
  regularization σ of the tests that produced it; `NaN` at the initial point. Empty for
  CondG and ProxGrad.
- `max_tr_B_trace::Vector{T}`: PDFPM only, `store_level` other than `:none`: per recorded point,
  `maxⱼ tr(Bⱼ)` of the curvature matrices of the subproblem that produced it; `NaN` at the
  initial point. Empty for CondG and ProxGrad.
- `sigma`, `max_tr_B`, `step_measure`, `step_threshold`, `required_decrease`, `actual_decrease`: PDFPM
  only, at any `store_level`: data of the last step test, namely σ, `T = maxⱼ tr(Bⱼ)`, `σ‖p-x‖`, the
  threshold of Step 3, the decrease required by Step 4 and `minⱼ[Fⱼ(x) - Fⱼ(p)]`. `NaN` before any
  test and for CondG and ProxGrad; `actual_decrease` is `NaN` if that test stopped at Step 3.
- `n_sigma_doublings::Int`: PDFPM only: total doublings of σ in the run.
- `status::Symbol`: Termination cause; see [`converged`](@ref).
- `message::String`: Human-readable termination or error message.
"""
mutable struct OptimResult{T<:AbstractFloat}
    x::Vector{T}                 # current point
    Fval::Vector{T}              # current F(x)
    x_init::Vector{T}            # initial point
    F_init::Vector{T}            # F(x_init)
    iter::Int
    abs_theta::T
    norm_progress::T
    n_f_evals::Int
    n_Jf_evals::Int
    total_time::Float64
    subproblem_time::Float64
    x_trace::Vector{Vector{T}}
    Fval_trace::Vector{Vector{T}}
    abs_theta_trace::Vector{T}
    norm_progress_trace::Vector{T}
    sigma_trace::Vector{T}
    max_tr_B_trace::Vector{T}
    sigma::T
    max_tr_B::T
    step_measure::T
    step_threshold::T
    required_decrease::T
    actual_decrease::T
    n_sigma_doublings::Int
    status::Symbol
    message::String
end

"""
    OptimResult(x0, Fval)

Construct an initial `OptimResult` with starting values.
"""
function OptimResult(x0::AbstractVector{T}, Fval::AbstractVector{T}) where {T<:AbstractFloat}
    return OptimResult{T}(
        copy(x0),               # x
        copy(Fval),             # Fval
        copy(x0),               # x_init
        copy(Fval),             # F_init
        0,                      # iter
        T(NaN),                 # abs_theta (not yet computed)
        T(NaN),                 # norm_progress
        0,                      # n_f_evals
        0,                      # n_Jf_evals
        0.0,                    # total_time
        0.0,                    # subproblem_time
        Vector{Vector{T}}(),    # x_trace
        Vector{Vector{T}}(),    # Fval_trace
        T[],                    # abs_theta_trace
        T[],                    # norm_progress_trace
        T[],                    # sigma_trace
        T[],                    # max_tr_B_trace
        T(NaN),                 # sigma
        T(NaN),                 # max_tr_B
        T(NaN),                 # step_measure
        T(NaN),                 # step_threshold
        T(NaN),                 # required_decrease
        T(NaN),                 # actual_decrease
        0,                      # n_sigma_doublings
        :not_started,           # status
        "Optimization not started")  # message
end

"""
    update!(result; kw...)

Update the `OptimResult` fields in-place.

# Keywords
- `x`: New iterate.
- `Fval`: New objective value.
- `abs_theta`: New optimality measure.
- `norm_progress`: New progress measure.
- `iter`: New iteration count.
- `sigma`, `max_tr_B`: PDFPM's σ and `maxⱼ tr(Bⱼ)` for the recorded point; they are only
  appended to `sigma_trace` and `max_tr_B_trace`. The scalar fields of the same names hold the
  last step test instead and are not set here.
- `store_level`: Trace level, one of `STORE_LEVELS`. Any level but `:none` appends `Fval`,
  `abs_theta`, `norm_progress`, `sigma` and `max_tr_B` to their traces; `:medium` and `:high`
  also append `x`.
"""
function update!(result::OptimResult{T}; x=nothing, Fval=nothing, abs_theta=nothing, norm_progress=nothing, iter=nothing, sigma=nothing, max_tr_B=nothing, store_level=:none) where {T<:AbstractFloat}
    store_low = store_level !== :none
    store_x = store_level === :medium || store_level === :high

    if x !== nothing
        result.x .= x
        if store_x
            push!(result.x_trace, copy(result.x))
        end
    end

    if Fval !== nothing
        result.Fval .= Fval
        if store_low
            push!(result.Fval_trace, copy(result.Fval))
        end
    end

    if abs_theta !== nothing
        result.abs_theta = abs_theta
        if store_low
            push!(result.abs_theta_trace, abs_theta)
        end
    end

    if norm_progress !== nothing
        result.norm_progress = norm_progress
        if store_low
            push!(result.norm_progress_trace, norm_progress)
        end
    end

    if store_low
        sigma !== nothing && push!(result.sigma_trace, sigma)
        max_tr_B !== nothing && push!(result.max_tr_B_trace, max_tr_B)
    end

    if iter !== nothing
        result.iter = iter
    end
end

"""
    struct SubProblemParams{T<:AbstractFloat}

Structure that stores the parameters of a subproblem.

# Fields
- `A::Vector{<:AbstractMatrix{T}}`: subproblem matrices `A`.
- `b::Vector{<:AbstractVector{T}}`: subproblem vectors `b`.
- `lb::AbstractVector{T}`: lower bounds of the variables.
- `ub::AbstractVector{T}`: upper bounds of the variables.
- `delta::T`: robustness parameter δ; `NaN` when built by the four-argument constructor,
  which takes arbitrary `b` instead.
- `h_factors`: one LU factorization of `Ãⱼ` per objective, used by [`evalh!`](@ref) for
  `hⱼ(x) = δ‖Ãⱼ⁻ᵀx‖₁`; `nothing` selects the linear-programming path.

Build it with [`robust_subproblem_params`](@ref) for the robust sets of the solvers. The
constructor `SubProblemParams(A, b, lb, ub)` keeps the linear-programming path for general
polyhedra `{z : Aⱼz ≤ bⱼ}`.
"""
struct SubProblemParams{T<:AbstractFloat}
    A::Vector{<:AbstractMatrix{T}}
    b::Vector{<:AbstractVector{T}}
    lb::AbstractVector{T}
    ub::AbstractVector{T}
    delta::T
    h_factors::Union{Nothing, Vector{LU{T, Matrix{T}, Vector{Int}}}}

    # Internal constructor with validations
    function SubProblemParams(A::Vector{<:AbstractMatrix{T}}, b::Vector{<:AbstractVector{T}}, lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:AbstractFloat}
        # Basic validations
        @assert length(A) == length(b) "Number of A matrices must equal number of b vectors"
        @assert length(lb) == length(ub) "Lower and upper bounds must have the same size"
        
        # Additional validation: check if dimensions are compatible
        if !isempty(A) && !isempty(b)
            n_vars = size(A[1], 2)  # number of columns of the first matrix A
            @assert length(lb) == n_vars "Dimension of bounds must equal number of variables"
            @assert length(ub) == n_vars "Dimension of bounds must equal number of variables"
            
            # Check if all matrices A have the same number of columns
            for (i, A_i) in enumerate(A)
                @assert size(A_i, 2) == n_vars "All A matrices must have the same number of columns (variables)"
            end
        end
        
        return new{T}(A, b, lb, ub, T(NaN), nothing)
    end

    # Used by robust_subproblem_params: same data plus δ and the factorizations of Ãⱼ.
    function SubProblemParams(params::SubProblemParams{T}, delta::T,
                              h_factors::Union{Nothing, Vector{LU{T, Matrix{T}, Vector{Int}}}}) where {T<:AbstractFloat}
        return new{T}(params.A, params.b, params.lb, params.ub, delta, h_factors)
    end
end

"""
    robust_subproblem_params(A, delta, lb, ub; max_cond_A=1e8)

Build the [`SubProblemParams`](@ref) of the robust sets
`Zⱼ = {z : -δe ≤ Ãⱼz ≤ δe}`, with `A[j] = Ãⱼ` square, as used by the three solvers.

Subproblems receive `Aⱼ = [Ãⱼ; -Ãⱼ]` and `bⱼ = δe`. For `δ > 0` each `Ãⱼ` is factorized once
and [`evalh!`](@ref) uses `hⱼ(x) = δ‖Ãⱼ⁻ᵀx‖₁`, without solving a linear program.

For `δ > 0`, each `Ãⱼ` is validated once, by its singular values: a non-square matrix, nonfinite
entries, `σ_min = 0` or `κ₂ > max_cond_A` raise `ArgumentError` with `j`, `κ₂` and the limit.
With `κ₂ ≈ 1e8` the relative error of the solve (`~κ₂u`) reaches the order of the error of the
linear program, which is why this is the default limit.

For `δ = 0` nothing is factorized and `h = 0`, the package's convention; this equals the
linear-programming definition only when `Ãⱼ` is invertible.
"""
function robust_subproblem_params(A::AbstractVector{<:AbstractMatrix}, delta::Real,
                                  lb::AbstractVector{T}, ub::AbstractVector{T};
                                  max_cond_A::Real=1e8) where {T<:AbstractFloat}
    delta_T = T(delta)
    isfinite(delta_T) && delta_T >= zero(T) || throw(ArgumentError("`delta` must be finite and nonnegative."))
    _check_max_cond_A(max_cond_A)
    A_T = [T.(Aj) for Aj in A]
    A2n = [[Aj; -Aj] for Aj in A_T]
    b2n = [delta_T * ones(T, 2 * size(Aj, 1)) for Aj in A_T]
    params = SubProblemParams(A2n, b2n, lb, ub)
    h_factors = iszero(delta_T) ? nothing : [_validated_h_factor(A_T[j], j, max_cond_A) for j in eachindex(A_T)]
    return SubProblemParams(params, delta_T, h_factors)
end

function _check_max_cond_A(max_cond_A::Real)
    !isnan(max_cond_A) && max_cond_A > 0 || throw(ArgumentError("`max_cond_A` must be positive."))
    return nothing
end

function _check_subproblem_tol(subproblem_tol)
    subproblem_tol === nothing && return nothing
    isfinite(subproblem_tol) && subproblem_tol > 0 || throw(ArgumentError("`subproblem_tol` must be finite and positive."))
    return nothing
end

function _validated_h_factor(Aj::AbstractMatrix{T}, j::Int, max_cond_A::Real) where {T<:AbstractFloat}
    size(Aj, 1) == size(Aj, 2) || throw(ArgumentError("A[$j] must be square; its size is $(size(Aj))."))
    (; kappa, sigma_min) = _h_matrix_diagnostics(Aj)
    limit = "limit `max_cond_A` = $max_cond_A"
    isnan(kappa) && throw(ArgumentError("A[$j] has nonfinite entries: κ₂ = $kappa; $limit."))
    iszero(sigma_min) && throw(ArgumentError("A[$j] is singular: κ₂ = $kappa; $limit."))
    kappa > max_cond_A && throw(ArgumentError("A[$j] is ill conditioned: κ₂ = $kappa; $limit."))
    F = lu(Matrix{T}(Aj); check=false)
    issuccess(F) || throw(ArgumentError("A[$j] is singular: its LU factorization has a zero pivot; κ₂ = $kappa; $limit."))
    return F
end

function _h_matrix_diagnostics(Aj::AbstractMatrix)
    T = float(eltype(Aj))
    all(isfinite, Aj) || return (; kappa=T(NaN), sigma_min=T(NaN))
    sigma = svdvals(Matrix{T}(Aj))
    sigma_min = sigma[end]
    # A zero matrix has σ_max = σ_min = 0: κ₂ is infinite, not 0/0.
    return (; kappa=iszero(sigma_min) ? T(Inf) : sigma[1] / sigma_min, sigma_min)
end

"""
    h_matrix_diagnostics(A)

Return, for each matrix in `A`, the named tuple `(; kappa, sigma_min)` with its 2-norm condition
number and smallest singular value, as validated by [`robust_subproblem_params`](@ref). Both are
`NaN` for nonfinite entries; a singular matrix has `kappa = Inf`. Nothing is rejected here.
"""
h_matrix_diagnostics(A::AbstractVector{<:AbstractMatrix}) = [_h_matrix_diagnostics(Aj) for Aj in A]



"""
    process_bounds(bound, n, default_value)

Helper function to expand scalar or `nothing` bounds into a vector of length `n`.
"""
function process_bounds(bound, n::Int, default_value)
    if bound === nothing
        return fill(default_value, n)
    elseif isa(bound, Number)
        T = typeof(default_value)
        return fill(convert(T, bound), n)
    else
        return bound
    end
end


"""
    bfgsupdate!(B, x, xprev, JF, JFprev, theta)

Perform a BFGS-like update in-place for a stack of matrices `B` (one for each objective).

Updates `B[j]` using the variation in `x` and the Jacobian `JF`.
The update is safeguarded to maintain positive definiteness where applicable.

# Arguments
- `B`: Vector of matrices `B_j`.
- `x`: Current point.
- `xprev`: Previous point.
- `JF`: Current Jacobian.
- `JFprev`: Previous Jacobian.
- `theta`: Optimality measure (used for scaling/safeguards).
"""
function bfgsupdate!(B::Vector{<:AbstractMatrix{T}}, x::AbstractVector{T},
                     xprev::AbstractVector{T},
                     JF::AbstractMatrix{T},
                     JFprev::AbstractMatrix{T},
                     theta::T) where {T<:Real}

    n = length(x)
    m, nJ = size(JF)
    @assert length(xprev) == n
    @assert size(JFprev) == (m, nJ)
    @assert nJ == n
    @assert length(B) == m
    @assert all(size(Bi) == (n, n) for Bi in B)

    s   = x .- xprev
    Dxs = maximum(JF * s)
    eps = T(1e-6) * min(abs(theta), one(theta))

    Bs = similar(s)
    y  = similar(s)

    @views for ind in 1:m
        yi  = view(JF,     ind, :)
        ypi = view(JFprev, ind, :)
        y .= yi .- ypi

        Bi = B[ind]
        mul!(Bs, Bi, s)

        sTy  = dot(s, y)
        sTBs = dot(s, Bs)

        # rho = Dxs - dot( row, s )  -- ensure Vector:
        rho = Dxs - dot(vec(ypi), s)

        if sTy > eps
            Bi .-= (Bs * transpose(Bs)) ./ sTBs
            Bi .+= (y  * transpose(y )) ./ sTy
        elseif rho > eps
            den = (rho - sTy)^2 + rho * sTBs
            BssT =  Bs * transpose(Bs)
            yyT  =  y  * transpose(y)
            ysTB =  y  * transpose(Bs)
            BsyT =  Bs * transpose(y)
            Bi .-= (rho/den)         .* BssT
            Bi .+= (sTBs/den)        .* yyT
            Bi .+= ((rho-sTy)/den)   .* (ysTB .+ BsyT)
        end
    end

    return B
end

"""
    jacobian_finite_diff!(J, f!, x; method=:central, λ=sqrt(eps(T)))

Fill the Jacobian `J` in-place using finite differences.

# Arguments
- `J::AbstractMatrix{T}`: Preallocated output Jacobian (m × n).
- `f!::Function`: Function to evaluate the objective `f(x) -> y`.
- `x::AbstractVector{T}`: Point where the Jacobian will be evaluated.
- `method::Symbol`: Finite difference method (`:forward`, `:backward`, `:central`).
- `λ::T`: Step size for finite differences.

# Returns
- `J`: The same matrix, filled in place. A column whose perturbed point rounds to `x`
  (`x[i] ± λ == x[i]`) is filled with `NaN` instead of a zero derivative.
"""
function jacobian_finite_diff!(J::AbstractMatrix{T}, f!::Function, x::AbstractVector{T}; method::Symbol=:central, λ::T=sqrt(eps(T))) where {T<:AbstractFloat}
    # Dimensions
    m, n = size(J)

    # Work buffers
    f_base  = zeros(T, m)
    f_plus  = similar(f_base)
    f_minus = similar(f_base)
    x_plus  = copy(x)
    x_minus = copy(x)

    # f(x) only for the one-sided differences; the central one does not use it
    method == :central || f!(f_base, x)

    if method == :forward
        for i in 1:n
            x_plus[i] += λ
            if x_plus[i] == x[i]
                J[:, i] .= T(NaN)
                continue
            end
            f!(f_plus, x_plus)
            @inbounds for j in 1:m
                J[j, i] = (f_plus[j] - f_base[j]) / λ
            end
            x_plus[i] = x[i]
        end

    elseif method == :backward
        for i in 1:n
            x_minus[i] -= λ
            if x_minus[i] == x[i]
                J[:, i] .= T(NaN)
                continue
            end
            f!(f_minus, x_minus)
            @inbounds for j in 1:m
                J[j, i] = (f_base[j] - f_minus[j]) / λ
            end
            x_minus[i] = x[i]
        end

    elseif method == :central
        half_inv = inv(T(2) * λ)
        for i in 1:n
            x_plus[i]  += λ
            x_minus[i] -= λ
            if x_plus[i] == x[i] || x_minus[i] == x[i]
                J[:, i] .= T(NaN)
                x_plus[i]  = x[i]
                x_minus[i] = x[i]
                continue
            end
            f!(f_plus,  x_plus)
            f!(f_minus, x_minus)
            @inbounds for j in 1:m
                J[j, i] = (f_plus[j] - f_minus[j]) * half_inv
            end
            x_plus[i]  = x[i]
            x_minus[i] = x[i]
        end
    else
        error("Invalid finite-difference method: $method. Choose :forward, :backward or :central.")
    end

    return J
end

"""
    noop_callback(args...)

Default callback that does nothing.
Returns `nothing`.
"""
function noop_callback(args...)
    # Does nothing
    return nothing
end
