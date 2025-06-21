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

mutable struct LineSearchResult{T<:AbstractFloat}
    stp::T # Step size
    F_final::AbstractVector{T} # Final function value
    nfev::Integer # Number of function evaluations
    success::Bool 
    message::String # Message
end

function LineSearchResult(; stp::T=0.0, F_final::AbstractVector{T}, nfev::Int=0, success::Bool=true, message::String="The Armijo line search has not started") where {T<:AbstractFloat}
    return LineSearchResult(stp, F_final, nfev, success, message)
end

function armijo(F::Function, x::AbstractVector{T}, d::AbstractVector{T}, Fx::AbstractVector{T}, 
                          Jfx::AbstractMatrix{T}; factor::T=nothing, options::ArmijoParams=ArmijoParams()) where {T<:AbstractFloat}
    
    
    result = LineSearchResult(F_final=Fx)
    
    m = length(Fx)

    function φ(α::T) where {T<:AbstractFloat}
        result.nfev += 1
        
        # Chamar F com o novo formato de retorno (valor, erro)
        F_val, error_msg = F(x + α * d)
        
        # Verificar se houve erro na avaliação
        if F_val === nothing
            # Propagar o erro para cima
            result.success = false
            result.message = error_msg
            # Retornar um valor que fará com que a condição de Armijo falhe
            # para garantir que o algoritmo pare
            return fill(T(Inf), m)
        end
        
        return F_val
    end

    φ_grad_0 = [dot(Jfx[i,:], d) for i in 1:m]

    if factor == nothing
        factor = maximum(φ_grad_0)
    end

    BacktrackMO!(result, φ, φ_grad_0, Fx, factor, options)
    
    if options.verbose >= 1
        @info result.message
    end

    return result
end

function safeguarded_backtrack(α::T, φ::T, φ_grad_0::T, φ_0::T; τ₁::T=0.05, τ₂::T=0.95) where {T<:AbstractFloat}

    """
    Calculates a new step `stpt` via quadratic interpolation,
    used in line search strategies with the Armijo condition.

    The idea is to model the unidimensional function φ(α) = f(x + αd) as a parabola:

        q(α) = f₀ + α g₀ + α² q₂

    where:
    - `f₀` is the function value at the current point (α = 0),
    - `g₀` is the directional derivative ∇f^T d at the current point,
    - `f` is the function value after a step `α` (current `stp`).

    Estimamos o coeficiente quadrático por:

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

        α_trial = (- φ_grad_0 * α^2) / (T(2.0) * (φ - φ_0 - α * φ_grad_0))

        if !(τ₁ * α <= α_trial <= τ₂ * α)
            α *= T(0.5)
        else
            α = α_trial
        end
    else
        α *= T(0.5)
    end

    return α
end

function BacktrackMO!(result::LineSearchResult, φ::Function, φ_grad_0::Vector{T}, φ_0::Vector{T}, factor::T, options::ArmijoParams) where {T<:AbstractFloat}

    if options.verbose >= 1
        @info "Starting Armijo line search"
    end

    #Check if the initial step size satisfies the Armijo condition
    φ_val = φ(options.alpha_init)
    α = options.alpha_init
    
    # Verificar se houve erro na avaliação
    if !result.success
        return result
    end
    
    failed_indices = findall(.!(φ_val .<= φ_0 .+ options.rho * α * factor))
    
    if isempty(failed_indices)
        result.stp = α
        result.message = "A step size that satisfies the Armijo condition was found"
        return result
    end

    for i in 1:options.max_iter

        for j in failed_indices
            α = safeguarded_backtrack(α, φ_val[j], φ_grad_0[j], φ_0[j])
            break
        end
        
        if α <= options.alpha_min
            result.stp = options.alpha_min
            result.message = "The Armijo line search did not converge"
            result.success = false
            return result
        end

        φ_val = φ(α)
        
        # Verificar se houve erro na avaliação
        if !result.success
            return result
        end
        
        failed_indices = findall(.!(φ_val .<= φ_0 .+ options.rho * α * factor))
        if isempty(failed_indices)
            result.stp = α
            result.F_final = φ_val
            result.message = "A step size that satisfies the Armijo condition was found"
            return result
        end
            
    end

    result.stp =  zero(T)
    result.F_final = φ_val
    result.message = "The maximum number of iterations for the Armijo line search was reached"
    result.success = false
    return result
end