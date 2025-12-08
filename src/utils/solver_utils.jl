"""
    OptimResult{T<:AbstractFloat}

Container structure holding the results produced by an optimization solver.

# Fields
- `x::Vector{T}`: Final iterate returned by the solver.
- `Fval::Vector{T}`: Value of the composite objective `F(x)` at `x`.
- `x_init::Vector{T}`: Initial point provided to the solver.
- `F_init::Vector{T}`: Value of the composite objective at `x_init`.
- `iter::Int`: Number of iterations performed.
- `abs_theta::T`: Absolute value of the optimality measure θ at the last iterate.
- `norm_progress::T`: Normalised ∞-norm progress between consecutive iterates.
- `n_f_evals::Int`: Number of objective evaluations (`f!`).
- `n_Jf_evals::Int`: Number of Jacobian evaluations (`Jf!`).
- `total_time::Float64`: Total wall-clock time spent by the solver (seconds).
- `subproblem_time::Float64`: Cumulative time spent solving sub-problems (seconds).
- `x_trace::Vector{Vector{T}}`: History of iterates (populated when `store_trace=true`).
- `Fval_trace::Vector{Vector{T}}`: History of `Fval` values when tracing is enabled.
- `abs_theta_trace::Vector{T}`: History of `abs_theta` values when tracing is enabled.
- `norm_progress_trace::Vector{T}`: History of `norm_progress` values when tracing is enabled.
- `success::Bool`: `true` if the solver terminated successfully, `false` otherwise.
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
    success::Bool
    message::String
end

function OptimResult(x0::AbstractVector{T}, Fval::AbstractVector{T}) where {T<:AbstractFloat}
    return OptimResult{T}(
        copy(x0),               # x
        copy(Fval),             # Fval
        copy(x0),               # x_init
        copy(Fval),             # F_init
        0,                      # iter
        T(Inf),                 # abs_theta
        T(NaN),                 # norm_progress
        0,                      # n_f_evals
        0,                      # n_Jf_evals
        0.0,                    # total_time
        0.0,                    # subproblem_time
        Vector{Vector{T}}(),    # x_trace
        Vector{Vector{T}}(),    # Fval_trace
        T[],                    # abs_theta_trace
        T[],                    # norm_progress_trace
        true,                   # success
        "Optimization not started")  # message
end

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
    SubProblemParams

Structure that stores the parameters of a subproblem.

# Fields
- `A::Vector{<:AbstractMatrix{T}}`: subproblem matrices A
- `b::Vector{<:AbstractVector{T}}`: subproblem vectors b
- `lb::AbstractVector{T}`: lower bounds of the variables
- `ub::AbstractVector{T}`: upper bounds of the variables
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

# -----------------------------------------------------------------------------
# Lightweight cache for evalh! models to avoid rebuilding HiGHS models every call
# Keyed by the SubProblemParams instance (object identity). Each entry stores a
# vector with (model, z) for each objective j.
const _EVALH_MODEL_CACHE = IdDict{SubProblemParams, Vector{NamedTuple{(:model, :z), Tuple{Model, Vector{JuMP.VariableRef}}}}}()

# Optional resource limits for HiGHS inside evalh!
struct _HiGHSLimits
    time_limit::Union{Nothing, Float64}
    simplex_iteration_limit::Union{Nothing, Int}
    ipm_iteration_limit::Union{Nothing, Int}
    output_flag::Union{Nothing, Bool}
end

const _EVALH_LIMITS = Base.RefValue{_HiGHSLimits}(_HiGHSLimits(nothing, nothing, nothing, nothing))

"""
    set_evalh_highs_limits!(; time_limit=nothing, simplex_iteration_limit=nothing, ipm_iteration_limit=nothing, output_flag=nothing)

Configure resource limits for HiGHS models used by `evalh!`.

All arguments are optional. Pass `nothing` to keep unset.
"""
function set_evalh_highs_limits!(; time_limit::Union{Nothing, Real}=nothing,
                                    simplex_iteration_limit::Union{Nothing, Integer}=nothing,
                                    ipm_iteration_limit::Union{Nothing, Integer}=nothing,
                                    output_flag::Union{Nothing, Bool}=nothing)
    _EVALH_LIMITS[] = _HiGHSLimits(
        time_limit === nothing ? nothing : Float64(time_limit),
        simplex_iteration_limit === nothing ? nothing : Int(simplex_iteration_limit),
        ipm_iteration_limit === nothing ? nothing : Int(ipm_iteration_limit),
        output_flag,
    )
    return nothing
end

function _ensure_h_models!(SubProbParams::SubProblemParams{T}, n::Int) where {T<:AbstractFloat}
    m = length(SubProbParams.A)
    cached = get(_EVALH_MODEL_CACHE, SubProbParams, nothing)
    # If missing or dimension mismatch, (re)build
    if cached === nothing || length(cached) != m || length(cached[1].z) != n
        models = Vector{NamedTuple{(:model, :z), Tuple{Model, Vector{JuMP.VariableRef}}}}(undef, m)
        for j in 1:m
            model = Model(HiGHS.Optimizer)
            set_silent(model)
            # Apply current resource limits (if any)
            lim = _EVALH_LIMITS[]
            if lim.time_limit !== nothing
                set_optimizer_attribute(model, "time_limit", lim.time_limit)
            end
            if lim.simplex_iteration_limit !== nothing
                set_optimizer_attribute(model, "simplex_iteration_limit", lim.simplex_iteration_limit)
            end
            if lim.ipm_iteration_limit !== nothing
                set_optimizer_attribute(model, "ipm_iteration_limit", lim.ipm_iteration_limit)
            end
            if lim.output_flag !== nothing
                set_optimizer_attribute(model, "output_flag", lim.output_flag)
            end
            @variable(model, z[1:n])
            @constraint(model, SubProbParams.A[j] * z .<= SubProbParams.b[j])
            models[j] = (model=model, z=z)
        end
        _EVALH_MODEL_CACHE[SubProbParams] = models
        return models
    else
        return cached
    end
end

"""
    clear_evalh_cache!()

Clear the cached HiGHS models used by `evalh!`.
Useful to release memory between runs with different problems.
"""
function clear_evalh_cache!()
    empty!(_EVALH_MODEL_CACHE)
    return nothing
end

function evalh!(h::AbstractVector{T}, x::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)
    
    # Check if any SubProbParams.b[j] has any zero value
    has_zero = false
    for j = 1:m
        if any(SubProbParams.b[j] .== 0)
            has_zero = true
            break
        end
    end
    
    # If any b[j] has a zero value, set all h to zero
    if has_zero
        fill!(h, zero(T))
        return true, "All h set to zero due to zero values in b"
    end
    
    # Build or reuse cached models
    models = _ensure_h_models!(SubProbParams, n)

    for j = 1:m
        mdl = models[j].model
        z   = models[j].z

        # Update objective: maximize dot(x, z)
        JuMP.set_objective(mdl, MOI.MAX_SENSE, dot(x, z))

        # Re-apply limits in case they were changed by the user after caching
        lim = _EVALH_LIMITS[]
        if lim.time_limit !== nothing
            set_optimizer_attribute(mdl, "time_limit", lim.time_limit)
        end
        if lim.simplex_iteration_limit !== nothing
            set_optimizer_attribute(mdl, "simplex_iteration_limit", lim.simplex_iteration_limit)
        end
        if lim.ipm_iteration_limit !== nothing
            set_optimizer_attribute(mdl, "ipm_iteration_limit", lim.ipm_iteration_limit)
        end
        if lim.output_flag !== nothing
            set_optimizer_attribute(mdl, "output_flag", lim.output_flag)
        end

        # Solve the problem
        optimize!(mdl)

        # Check if solution was found (accept multiple success statuses)
        ts = termination_status(mdl)
        if ts == MOI.OPTIMAL || ts == MOI.LOCALLY_SOLVED || ts == MOI.ALMOST_OPTIMAL || ts == MOI.ALMOST_LOCALLY_SOLVED
            h[j] = objective_value(mdl)
        else
            @warn "ERROR: Linear program could not be solved! Status: $(ts)"
            return false, "Failed evaluating h_$j(x^k)."
        end
    end

    return true, "All subproblems solved successfully"
end

# Helper function to process bounds
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

function print_table(result::OptimResult)
    if result.iter == 0
        separator = "-"^70
        @info separator
        headers = ["Iter", "|θ(x^k)|", "    ||x^k - x^{k-1}||/max{1, ||x^{k-1}||}"]
        n_fields = length(headers)
        format_str = "%7s  %14s  %25s"
        fmt = Printf.Format(format_str)
        header_str = Printf.format(fmt, headers...)
        
        @info header_str
        @info separator
        
        # Print progress row
        progress_str = @sprintf("%5d  %18.6e  %25s", 
                        result.iter, result.abs_theta, "--")
        @info progress_str
    else
        # Print progress row
        progress_str = @sprintf("%5d  %18.6e  %25.6e", 
                result.iter, result.abs_theta, result.norm_progress)
        @info progress_str
    end
end

"""
BFGS-like update for a *stack* of matrices B[:, :, j], j=1..m (in-place).
- Expected formats:
    length(x) = length(xprev) = n
    size(JF) = size(JFprev) = (m, n)
    size(B[j]) = (n, n) for all j
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

# Arguments:
- `J::AbstractMatrix{T}`: Preallocated output Jacobian (m × n)
- `f!::Function`: Function to evaluate the objective f(x): R^n -> R^m
- `x::AbstractVector{T}`: Point where the Jacobian will be evaluated
- `method::Symbol`: Finite difference method (:forward, :backward, :central)
- `λ::T`: Step size for finite differences

# Returns:
- `J` (the same matrix, filled in place)
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
    jacobian_finite_diff(f!, x, m; method=:central, λ=sqrt(eps(T)))

Allocate and return a new Jacobian (kept for compatibility). Prefer `jacobian_finite_diff!`.
"""
function jacobian_finite_diff(f!::Function, x::AbstractVector{T}, m::Int; method::Symbol=:central, λ::T=sqrt(eps(T))) where {T<:AbstractFloat}
    n = length(x)
    J = zeros(T, m, n)
    return jacobian_finite_diff!(J, f!, x; method=method, λ=λ)
end

#TODO: Use this function to check if the subproblem is feasible# Optional: Presolve function to check problem feasibility
function check_subproblem_feasibility(x::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}
    n = length(x)
    
    # Check if x is within bounds
    if any(x .< SubProbParams.lb) || any(x .> SubProbParams.ub)
        return false, "Initial point violates bounds"
    end
    
    # Check if constraint system has a solution
    for A in SubProbParams.A
        if size(A, 2) != n
            return false, "Dimension mismatch in constraint matrix"
        end
    end
    
    return true, "Problem appears feasible"
end

# Optional: Helper function to create a solver cache
function create_pdfpm_cache(::Type{T}) where {T<:AbstractFloat}
    model = Model(Ipopt.Optimizer)
    return model
end

"""
    noop_callback

Default callback that does nothing.
"""
function noop_callback(args...)
    # Does nothing
    return nothing
end
