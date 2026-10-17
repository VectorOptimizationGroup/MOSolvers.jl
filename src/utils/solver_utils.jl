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
- `x_trace::Vector{Vector{T}}`: History of iterates (populated when `store_trace=true`).
- `Fval_trace::Vector{Vector{T}}`: History of `Fval` values when tracing is enabled.
- `abs_theta_trace::Vector{T}`: History of `abs_theta` values when tracing is enabled.
- `norm_progress_trace::Vector{T}`: History of `norm_progress` values when tracing is enabled.
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
- `store_trace`: If `true`, appends `x`, `Fval`, `abs_theta`, etc. to trace history.
"""
function update!(result::OptimResult{T}; x=nothing, Fval=nothing, abs_theta=nothing, norm_progress=nothing, iter=nothing, store_trace=false) where {T<:AbstractFloat}
    if x !== nothing
        result.x .= x
        if store_trace
            push!(result.x_trace, copy(result.x))
        end
    end

    if Fval !== nothing
        result.Fval .= Fval
        if store_trace
            push!(result.Fval_trace, copy(result.Fval))
        end
    end

    if abs_theta !== nothing
        result.abs_theta = abs_theta
        if store_trace
            push!(result.abs_theta_trace, abs_theta)
        end
    end

    if norm_progress !== nothing
        result.norm_progress = norm_progress
        if store_trace
            push!(result.norm_progress_trace, norm_progress)
        end
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
"""
struct SubProblemParams{T<:AbstractFloat}
    A::Vector{<:AbstractMatrix{T}}
    b::Vector{<:AbstractVector{T}}
    lb::AbstractVector{T}
    ub::AbstractVector{T}
    
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
        
        return new{T}(A, b, lb, ub)
    end
end



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
- `J`: The same matrix, filled in place.
"""
function jacobian_finite_diff!(J::AbstractMatrix{T}, f!::Function, x::AbstractVector{T}; method::Symbol=:central, λ::T=sqrt(eps(T))) where {T<:AbstractFloat}
    # Dimensions
    m, n = size(J)

    # Base evaluation f(x)
    f_base = zeros(T, m)
    f!(f_base, x)

    # Work buffers
    f_plus  = similar(f_base)
    f_minus = similar(f_base)
    x_plus  = copy(x)
    x_minus = copy(x)

    if method == :forward
        for i in 1:n
            x_plus[i] += λ
            f!(f_plus, x_plus)
            @inbounds for j in 1:m
                J[j, i] = (f_plus[j] - f_base[j]) / λ
            end
            x_plus[i] = x[i]
        end

    elseif method == :backward
        for i in 1:n
            x_minus[i] -= λ
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
