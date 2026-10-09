"""
The subroutine is designed to find a positive step-size that satisfies the sufficient decrease condition (Armijo condition).

    Multi-Objective Armijo Line Search

Let F: ℝⁿ → ℝᵐ, F(x) = (F₁(x), ..., Fₘ(x)), be a continuous multi-objective 
differentiable function. If d ∈ ℝⁿ is a descent direction of F at x, then taking 
φᵢ: ℝ → ℝ for i = 1,...,m, where φᵢ(α) := Fᵢ(x + α * d), for all i = 1,...,m, 
the sufficient decrease condition:

    φᵢ(α) ≤ φᵢ(0) + ρ * α * ANYTHING(x) for all i = 1,...,m (Armijo condition)

corresponds to the multi-objective Armijo condition for the step-size α.

Here, ANYTHING(x) is a function of x that is used to define the sufficient decrease condition.
For example:
    • ANYTHING(x) = JF(x) * d # Where JF(x) is the Jacobian of F at x # See [1]
    • ANYTHING(x) = max{∇φᵢ(0) : i ∈ {1,...,m}} = max{∇Fᵢ(x)ᵀ * d : i ∈ {1,...,m}} # See [2]
    • ANYTHING(x) = -|θ(x)| # Where θ(x) is the optimal value of the subproblem associated with finding the descent direction d # See [3]
    
See the following references for more details about choosing the ANYTHING(x) function:
    • [1] Fliege, J.; Svaiter, B. F. Steepest descent methods for multicriteria 
      optimization. Mathematical Methods of Operations Research, 51(3):479–494, 2000.
    • [2] Lucambio Pérez, L. R.; Prudente, L. F. Nonlinear Conjugate Gradient 
      Methods for Vector Optimization. SIAM J. Optim., 28(3):2690–2720, 2018.
    • [3] Assunção, P. B., Ferreira, O. P., & Prudente, L. F. (2023). A generalized 
      conditional gradient method for multiobjective composite optimization problems. 
      Optimization, 74(2), 473–503.
"""

"""
    mutable struct ArmijoParams{T<:AbstractFloat}

Parameters for the Armijo line search.

# Fields
- `verbose::Integer`: Verbosity level (0: none, 1: info).
- `max_iter::Int`: Maximum number of backtracking iterations.
- `rho::T`: Sufficient decrease factor (0 < rho < 1).
- `alpha_init::T`: Initial step size.
- `alpha_max::T`: Maximum allowed step size.
- `alpha_min::T`: Minimum allowed step size.
- `print_interval::Int`: Interval for printing progress (if verbose).
- `store_trace::Bool`: Whether to store trace information (not fully used in simple backtracking).
"""
mutable struct ArmijoParams{T<:AbstractFloat}
    verbose::Integer # Verbose level
    max_iter::Int # Maximum number of iterations
    rho::T # Reduction factor
    alpha_init::T # Initial step size
    alpha_max::T # Maximum step size
    alpha_min::T # Minimum step size
    print_interval::Int # Print interval
    store_trace::Bool # Store trace 
end

"""
    ArmijoParams(; kw...)

Construct `ArmijoParams` with validated defaults.

# Keywords
- `verbose::Integer=0`: Verbose level.
- `max_iter::Integer=1000`: Max iterations.
- `rho::Real=1e-4`: Sufficient decrease factor.
- `alpha_init::Real=1.0`: Initial step size.
- `alpha_max::Real=1.0`: Max step size.
- `alpha_min::Real=1e-15`: Min step size.
- `print_interval::Integer=10`: Print interval.
- `store_trace::Bool=false`: Store trace.
"""
function ArmijoParams(; verbose=0, max_iter=1000, rho=1e-4, alpha_init=1.0, alpha_max=1.0, alpha_min=1e-15, print_interval=10, store_trace=false)

    if rho <= 0
        error("The reduction factor rho must be positive")
    end
    if alpha_init < alpha_min || alpha_max < alpha_init
        error("The initial step size alpha_init must belong to the interval [alpha_min, alpha_max]")
    end
    if alpha_max <= alpha_min
        error("The maximum step size alpha_max must be greater than the minimum step size alpha_min")
    end
    if alpha_min <= 0 || alpha_max <= 0
        error("The step sizes alpha_min and alpha_max must be positive")
    end

    return ArmijoParams(verbose, max_iter, rho, alpha_init, alpha_max, alpha_min, print_interval, store_trace)
end

"""
    mutable struct LineSearchResult{T<:AbstractFloat}

Result of a line search procedure.

# Fields
- `stp::T`: Accepted step size.
- `F_final::AbstractVector{T}`: Function value at the accepted step.
- `nfev::Integer`: Number of function evaluations performed.
- `success::Bool`: `true` if the line search converged.
- `message::String`: Status message.
"""
mutable struct LineSearchResult{T<:AbstractFloat}
    stp::T # Step size
    F_final::AbstractVector{T} # Final function value
    nfev::Integer # Number of function evaluations
    success::Bool 
    message::String # Message
end

function LineSearchResult(; F_final::AbstractVector{T}, stp::T=zero(eltype(F_final)), nfev::Int=0, success::Bool=true, message::String="The Armijo line search has not started") where {T<:AbstractFloat}
    return LineSearchResult(stp, copy(F_final), nfev, success, message)
end

"""
    armijo(F, x, d, Fx, Jfx; factor=nothing, options=ArmijoParams())

Perform a Multiobjective Armijo backtracking line search.

Finds a step size `α` such that:
`F_i(x + α d) ≤ F_i(x) + ρ * α * factor` for all `i`.
Nonfinite trial objectives trigger step halving. Explicit evaluation errors,
exhausted reduction limits, and steps that round back to `x` terminate with failure.

# Arguments
- `F::Function`: Objective function `F(x)`. Must return `(value, error_msg)`.
- `x::AbstractVector`: Current point.
- `d::AbstractVector`: Descent direction.
- `Fx::AbstractVector`: Value of `F` at `x`.
- `Jfx::AbstractMatrix`: Jacobian of `F` at `x` (used if `factor` is nothing).

# Keywords
- `factor::Union{Real, Nothing}`: The term used in the sufficient decrease condition.
    - If `nothing`, uses `maximum(Jfx * d)`.
    - Can be a scalar (e.g., `-abs(θ)` from a subproblem).
- `options::ArmijoParams`: Line search parameters.

"""
function armijo(F::Function, x::AbstractVector{T}, d::AbstractVector{T}, Fx::AbstractVector{T},
                Jfx::AbstractMatrix{T}; factor::Union{T,Nothing}=nothing,
                options::ArmijoParams=ArmijoParams()) where {T<:AbstractFloat}
    
    
    result = LineSearchResult(F_final=Fx)

    m = length(Fx)

    # Typed local copies of options
    ρ      = T(options.rho)
    α_init = T(options.alpha_init)
    α_min  = T(options.alpha_min)
    α_max  = T(options.alpha_max)

    # Reusable trial point buffer to avoid allocations: y = x + α d
    y = similar(x)

    function φ(αk::T)
        @inbounds begin
            @. y = x + αk * d
        end
        if y == x
            result.success = false
            result.message = "The Armijo trial point is indistinguishable from x"
            return fill(T(Inf), m)
        end
        # Call F with the new return format (value, error)
        result.nfev += 1
        F_val, error_msg = F(y)
        
        # Check if there was an error in evaluation
        if F_val === nothing
            # Propagate the error up
            result.success = false
            result.message = error_msg
            # Return a value that will cause the Armijo condition to fail
            # to ensure the algorithm stops
            return fill(T(Inf), m)
        end
        
        return F_val
    end

    @views φ_grad_0 = [dot(view(Jfx, i, :), d) for i in 1:m]

    factor_val = factor === nothing ? maximum(φ_grad_0) : factor
    factor_val = T(factor_val)

    # Clamp initial step into [α_min, α_max]
    if !(α_min <= α_init <= α_max)
        α_init = clamp(α_init, α_min, α_max)
    end

    BacktrackMO!(result, φ, φ_grad_0, Fx, factor_val, options)
    
    if options.verbose >= 1
        @info result.message
    end

    return result
end

"""
    safeguarded_backtrack(α, φ, φ_grad_0, φ_0; τ₁=0.05, τ₂=0.95)

Calculate a new step size via quadratic interpolation with safeguards.

Models `φ(α)` as a quadratic `q(α)` matched at `α=0` and `α=current`.
Result is clamped to `[τ₁ * α, τ₂ * α]` to ensure robust decrease.
"""
function safeguarded_backtrack(α::T, φ::T, φ_grad_0::T, φ_0::T; τ₁::T=T(0.05), τ₂::T=T(0.95)) where {T<:AbstractFloat}

    """
    Calculates a new step `stpt` via quadratic interpolation,
    used in line search strategies with the Armijo condition.

    The idea is to model the unidimensional function φ(α) = f(x + αd) as a parabola:

        q(α) = f₀ + α g₀ + α² q₂

    where:
    - `f₀` is the function value at the current point (α = 0),
    - `g₀` is the directional derivative ∇f^T d at the current point,
    - `f` is the function value after a step `α` (current `stp`).

    We estimate the quadratic coefficient by:

        q₂ = (f - f₀ - α g₀) / α²

    and the minimum of the parabola (new candidate step) is given by:

        stpt = (- g₀ * α²) / (2.0 * (φ(α) - φ₀ - α * φ_grad_0))

    This step is a prediction of where the function can reach a minimum
    based on three points: f₀, f and g₀. However, as the quadratic interpolation
    can generate unreliable steps (too large or too small),
    we apply a safeguard criterion.

    For this, we impose that the new step `stpt` belongs to the interval:

        stpt ∈ [τ₁ * α, τ₂ * α], with 0 < τ₁ < τ₂ < 1

    This interval proportional to the previous step (`α`) guarantees that:
    - The new step is not too small to compromise the progress;
    - Not too large to extrapolate excessively the data used in the interpolation;
    - The algorithm maintains numerical robustness and stability.

    If `stpt` falls outside this interval, we opt for a more conservative strategy,
    such as the bisection of the search interval.
"""
    if φ_grad_0 < zero(T)
        den = (φ - φ_0 - α * φ_grad_0)
        if iszero(den) || !isfinite(den)
            α *= T(0.5)
        else
            α_trial = (- φ_grad_0 * α^2) / (T(2.0) * den)
            if !(τ₁ * α <= α_trial <= τ₂ * α) || !isfinite(α_trial)
                α *= T(0.5)
            else
                α = α_trial
            end
        end
    else
        α *= T(0.5)
    end

    return α
end

"""
    BacktrackMO!(result, φ, φ_grad_0, φ_0, factor, options)

Execute backtracking loop to satisfy the Armijo condition.
Updates `result` in place.
"""
function BacktrackMO!(result::LineSearchResult{T}, φ::Function, φ_grad_0::Vector{T}, φ_0::Vector{T}, factor::T, options::ArmijoParams) where {T<:AbstractFloat}

    if options.verbose >= 1
        @info "Starting Armijo line search"
    end

    # Typed local copies of options (keep in T)
    ρ      = T(options.rho)
    α      = T(options.alpha_init)
    α_min  = T(options.alpha_min)
    α_max  = T(options.alpha_max)
    α      = clamp(α, α_min, α_max)
    
    # Check if the initial step size satisfies the Armijo condition
    φ_val = φ(α)
    
    # Check if there was an error in evaluation
    if !result.success
        return result
    end
    
    failed_indices = findall(@. !isfinite(φ_val) | !(φ_val <= (φ_0 + ρ * α * factor)))
    
    if isempty(failed_indices)
        result.stp = α
        result.F_final = copy(φ_val)
        result.success = true
        result.message = "A step size that satisfies the Armijo condition was found"
        return result
    end

    for i in 1:options.max_iter

        if any(!isfinite, φ_val)
            # Interpolation is undefined for nonfinite objective values.
            α *= T(0.5)
        else
            j = first(failed_indices)
            α = safeguarded_backtrack(α, φ_val[j], φ_grad_0[j], φ_0[j])
        end
        
        if α <= α_min
            result.stp = α_min
            result.message = "The Armijo line search did not converge"
            result.success = false
            return result
        end

        φ_val = φ(α)
        
        # Check if there was an error in evaluation
        if !result.success
            return result
        end
        
        failed_indices = findall(@. !isfinite(φ_val) | !(φ_val <= (φ_0 + ρ * α * factor)))
        if isempty(failed_indices)
            result.stp = α
            result.F_final = copy(φ_val)
            result.success = true
            result.message = "A step size that satisfies the Armijo condition was found"
            return result
        end
            
    end

    result.stp =  zero(T)
    result.F_final = copy(φ_val)
    result.message = "The maximum number of iterations for the Armijo line search was reached"
    result.success = false
    return result
end
