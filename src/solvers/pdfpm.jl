mutable struct PDFPMParams{T<:AbstractFloat}
    verbose::Integer # Verbose level
    max_iter::Integer # Maximum number of iterations
    opt_tol::T # Optimality tolerance
    ftol::T # Function tolerance
    max_time::T # Maximum time in seconds
    print_interval::Integer # Print interval
    stop_criteria::Symbol # Additional stop criteria
    store_trace::Bool # Store trace
    epsilon::T # Epsilon parameter for finite differences
    sigma::T # Sigma parameter for regularization
    alpha::T # Alpha parameter for sufficient decrease
    max_subproblem_iter::Integer # Maximum subproblem iterations
    lambda_factor::Union{Function, T} # Percentage in [0,1] of the interval [0, ϵ/(σ√n)] used as λ via λ = max(lambda_factor * ϵ/(σ√n), λ_min). If a Function, it is evaluated at runtime; recommended signature: lambda_factor(; iter, sigma, n) -> Real in [0,1]
    lambda_min::T # Minimum lambda for finite differences
end


"""	
    PDFPM_options(; verbose::Integer = 1, max_iter::Integer = 1000, opt_tol::T = 1e-4, ftol::T = 1e-4, max_time::T = 3600.0, print_interval::Integer = 10, store_trace::Bool = false, epsilon::T = 1e-10, sigma::T = 1.0, alpha::T = 0.1, max_subproblem_iter::Integer = 200, lambda_factor::T = 0.1, lambda_min::T = 1e-12)

# Arguments:
# - `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed)
# - `max_iter::Integer = 1000`: Maximum number of iterations
# - `opt_tol::T = 1e-4`: Optimality tolerance
# - `ftol::T = 1e-4`: Function tolerance
# - `max_time::T = 3600.0`: Maximum time in seconds
# - `print_interval::Integer = 10`: Print interval
# - `stop_criteria::Symbol = :pdfpm`:  Additional stop criteria (:condg, :proxgrad, :pdfpm)
# - `store_trace::Bool = false`: Whether to store the trace of x during optimization
# - `epsilon::T = 1e-10`: Epsilon parameter for finite differences
# - `sigma::T = 1.0`: Sigma parameter for regularization
# - `alpha::T = 0.1`: Alpha parameter for sufficient decrease
# - `max_subproblem_iter::Integer = 200`: Maximum subproblem iterations
# - `lambda_factor::Union{T,Function} = 0.1`: If a number, it is clamped to [0,1] and used as a percentage of the interval [0, ϵ/(σ√n)] to define λ via
#     λ = max(lambda_factor * ϵ/(σ√n), λ_min).
#   If a function, it is evaluated at runtime (per iteration) and should return a number in [0,1].
#   Recommended function signature:
#     lambda_factor(; iter, sigma, n) -> Real in [0,1]
#   Current implementation calls it at least with `iter` as keyword: lambda_factor(; iter=iter).
# - `lambda_min::T = 1e-12`: Minimum lambda for finite differences (λ is floored by this value).
"""
function PDFPM_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=10, stop_criteria=:pdfpm, store_trace=false, epsilon=1e-10, sigma=1.0, alpha=0.1, max_subproblem_iter=200, lambda_factor=0.1, lambda_min=1e-12)
    if stop_criteria ∉ [:pdfpm, :NormProgress]
        error("The stop criteria is not supported: $stop_criteria")
    end

    # Do not evaluate lambda_factor if it's a function; evaluation happens during optimize
    if !(lambda_factor isa Function)
        # Clamp numeric lambda_factor into [0, 1]
        if !(lambda_factor isa Real)
            @warn "lambda_factor must be a number in [0,1] or a function; using 0.1"
            lambda_factor = 0.1
        else
            if !isfinite(lambda_factor)
                @warn "lambda_factor is not finite; using 0.1"
                lambda_factor = 0.1
            end
            lambda_factor = clamp(lambda_factor, 0.0, 1.0)
        end
    end

    return PDFPMParams(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_trace, epsilon, sigma, alpha, max_subproblem_iter, lambda_factor, lambda_min)
end

"""
    PDFPM(evalf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, x0::AbstractVector{T}, options::PDFPMParams=PDFPM_options(); evalJf::Union{Nothing,Function}=nothing, lb::Union{Nothing,AbstractVector{T},Number}=nothing, ub::Union{Nothing,AbstractVector{T},Number}=nothing, callback=noop_callback) where {T<:AbstractFloat}

# Arguments:
# - `evalf::Function`: Function to evaluate the objective function f(x): R^n -> R^m
# - `A::Vector{<:AbstractMatrix{T}}`: Vector of matrices A for each objective
# - `delta::T`: Uncertainty parameter
# - `x0::AbstractVector{T}`: Initial guess
# - `options::PDFPMParams=PDFPM_options()`: Options for the PDFPM solver
# - `evalJf::Union{Nothing,Function}=nothing`: Function to evaluate the Jacobian of the objective function Jf(x): R^n -> R^{m×n} (optional, uses finite differences if not provided)
# - `lb::Union{Nothing,AbstractVector{T},Number}=nothing`: Lower bounds
# - `ub::Union{Nothing,AbstractVector{T},Number}=nothing`: Upper bounds
# - `callback::Function=noop_callback`: Optional hook executed every iteration, with signature `callback(iter, result) -> Bool`. Return `true` to stop; any other value continues. Use for logging or custom stopping rules.
"""

function PDFPM(evalf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, 
    x0::AbstractVector{T}, options::PDFPMParams=PDFPM_options(); 
    evalJf::Union{Nothing,Function}=nothing,
    lb::Union{Nothing,AbstractVector{T},Number}=nothing, 
    ub::Union{Nothing,AbstractVector{T},Number}=nothing,
    callback=noop_callback) where {T<:AbstractFloat}

    start_time = time()

    n = length(x0)
    m = length(A)
    A2n = [[A[i]; -A[i]] for i in 1:m]
    b2n = [delta * ones(2*n) for i in 1:m]

    # Validations
    @assert m > 0 "At least one matrix A must be provided"

    for i in 1:m
        @assert size(A[i], 1) == n && size(A[i], 2) == n "Matrix A[$i] must be square of size $n×$n"
    end

    # Process bounds
    lb_vec = process_bounds(lb, n, T(-Inf))
    ub_vec = process_bounds(ub, n, T(Inf))

    # Check if lb <= ub
    for i in 1:n
        @assert lb_vec[i] <= ub_vec[i] "Lower bound lb[$i] = $(lb_vec[i]) must be less than or equal to upper bound ub[$i] = $(ub_vec[i])"
    end

    # Check if x0 is within bounds
    for i in 1:n
        if !(lb_vec[i] <= x0[i] <= ub_vec[i])
            @warn "x0[$i] = $(x0[i]) is out of bounds [$(lb_vec[i]), $(ub_vec[i])]"
        end
    end

    # Create the structure internally
    SubProbParams = SubProblemParams(A2n, b2n, lb_vec, ub_vec)

    # Create result structure
    result = OptimResult(x0, zeros(T, m))

    function f!(f, x)
        result.n_f_evals += 1
        f .= evalf(x)
        return f
    end

    Jf_is_analytic = evalJf !== nothing

    # Create Jf! function based on whether evalJf was provided or not
    function Jf!(Jf, x; method=:central, lambda=sqrt(eps(T)))
        result.n_Jf_evals += 1
        if evalJf !== nothing
            Jf .= evalJf(x)
        else
            # Use finite differences
            jacobian_finite_diff!(Jf, f!, x; method=method, λ=lambda)
        end
        return Jf
    end

    """	
    Create the function
                    h(x): R^n -> R^m,
                    h(x) = (h_1(x), ..., h_m(x)),
    where h_j(x) := max{<x,z> | z ∈ Z_j} and Z_j := {z ∈ R^n | -δ·e ≤ A_j z ≤ δ·e}  for j = 1, ..., m
    """

    function h!(h, x)
        success, message = evalh!(h, x, SubProbParams)
        if !success
            result.success = false
            result.message = "evalh! failed: " * message
        end
        return h
    end

    function subproblem(x, JF, h, method::Symbol; sigma::T=1.0, B::Union{Nothing, Vector{<:AbstractMatrix{T}}}=nothing)
        if method == :condg
            theta, p, success, message = subproblem_condg(x, JF, h, SubProbParams; verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of CondG failed: " * message
            end
            return theta, p
        elseif method == :proxgrad
            theta, p, success, message = subproblem_proxgrad(x, JF, h, SubProbParams; sigma=sigma, verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of ProxGrad failed: " * message
            end
            return theta, p
        elseif method == :pdfpm
            theta, p, success, message = subproblem_pdfpm(x, JF, h, SubProbParams; sigma=sigma, B=B, verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of PDFPM failed: " * message
            end
            return theta, p
        else
            error("The subproblem method is not supported: $method")
        end
    end

    _PDFPM_optimize!(result, f!, Jf!, Jf_is_analytic, h!, m, subproblem, options, callback)
    result.total_time = time() - start_time
    return result
end

"""
    _PDFPM_optimize!(f!, Jf!, h!, x0, options, callback, result)

Internal function that implements the PDFPM algorithm.
"""
function _PDFPM_optimize!(result::OptimResult, f!::Function, Jf!::Function, Jf_is_analytic::Bool, h!::Function, m::Int, subproblem::Function, options::PDFPMParams, callback::Function)
    
    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    T = eltype(x)
    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    JF_prev = similar(JF)
    h = zeros(T, m)
    h_prev = zeros(T, m)
    #Initialize theta
    theta = T(NaN)

    # Initialize B matrices as identity for each objective
    B = [Matrix{T}(I, n, n) for _ in 1:m]
    
    # Initialize sigma
    sigma = options.sigma

    # Initialize lambda (keep computations in type T)
    sqrt_n = sqrt(T(n))
    scale = options.epsilon / (options.sigma * sqrt_n)
    lambda_factor_val = options.lambda_factor isa Function ? convert(T, options.lambda_factor(; iter=0)) : convert(T, options.lambda_factor)

    lambda = ifelse(
        isfinite(scale) && scale > 0,
        max(scale * lambda_factor_val, options.lambda_min),
        options.lambda_min
    )

    # Start timing to check for maximum time
    start_time = time()
    
    if options.verbose > 0
        @info "----------------------------------------------------------------------"
        @info "A Partially Derivative-Free Proximal Method for Composite Multiojective \n Optimization in the Hölder Setting"
        @info "----------------------------------------------------------------------"
        @info "N of variables=$(n), N of objectives=$(m), Opt tol=$(options.opt_tol), Max iter=$(options.max_iter)"
        @info "Epsilon=$(options.epsilon), Sigma=$(sigma), Alpha=$(options.alpha)"
    end

    # Compute F and JF
    f!(Fx, x)
    Jf!(JF, x; lambda=lambda)
    h!(h, x)

    # Check if h! failed
    if !result.success
        result.message = "h! failed"
        return result
    end

    h_prev .= h

    # Solve the subproblem
    start_subproblem_time = time()
    theta, p = subproblem(x, JF, h, :pdfpm; sigma=sigma, B=B)
    result.subproblem_time += time() - start_subproblem_time
    if !result.success
        return result
    end

    # Update result
    update!(result, x=x, abs_theta=abs(theta), Fval=Fx + h, norm_progress=T(NaN), store_trace=options.store_trace)
    result.F_init .= result.Fval

    if options.verbose > 0
        if result.iter == 0 || (result.iter % options.print_interval == 0)
            print_table(result)
        end
    end

    # Stopping criteria
    if result.abs_theta <= options.opt_tol
        result.success = true
        result.message = "Solution found at initial point with |θ_PDFPM| ≤ $(options.opt_tol)"
        return result
    end

    for iter in 1:options.max_iter

        if time() - start_time >= options.max_time
            result.success = false
            result.message = "Maximum time reached $(options.max_time)"
            return result
        end
        # Store previous value of h
        h_prev .= h
        
        for sub_iter in 1:options.max_subproblem_iter
            # Check convergence
            if sigma * norm(p - x) <= options.epsilon
                x .= p
                result.success = true
                result.message = "Solution was found"
                # Update result
                norm_progress = T(norm(x - result.x, Inf) / max(1, norm(result.x, Inf)))
                # Compute F in-place: Fx = f(x) + h(x)
                f!(Fx, x)
                h!(h, x)
                # Check if h! failed in F evaluation
                if !result.success
                    return result
                end
                Fx .+= h
                #  WARNING: abs_theta is not updated here.
                update!(result, x=x, Fval=Fx, abs_theta=zero(T), norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)
                if options.verbose > 0
                    if iter == 0 || (iter % options.print_interval == 0)
                        print_table(result)
                    end
                    @info "Converged at iteration $iter with small tolerance"
                end

                return result
            end

            # Check sufficient decrease
            f!(Fx, p)
            h!(h, p)

            # Check if h! failed in F evaluation
            if !result.success
                return result
            end

            Fx .+= h

            if all(Fx .<= result.Fval .- (options.alpha * options.epsilon^2) / (2 * sigma))
                # Update x
                x .= p
                break
            else
                if sub_iter >= options.max_subproblem_iter
                    result.success = false
                    result.message = "The maximum subiteration limit was reached"
                    return result
                end
                sigma = sigma * 2
                if !Jf_is_analytic
                    # Update lambda (keep computations in type T)
                    scale = options.epsilon / (sigma * sqrt_n)
                    lambda_factor_val = options.lambda_factor isa Function ? convert(T, options.lambda_factor(; iter=iter)) : convert(T, options.lambda_factor)
                    lambda = ifelse(
                        isfinite(scale) && scale > 0,
                        max(scale * lambda_factor_val, options.lambda_min),
                        options.lambda_min
                    )
                    Jf!(JF, x; lambda=lambda)
                end
            end
            # Check maximum time
            if time() - start_time >= options.max_time
                result.success = false
                result.message = "Maximum time reached $(options.max_time)"
                return result
            end
            # Solve the subproblem
            start_subproblem_time = time()
            theta, p = subproblem(x, JF, h_prev, :pdfpm; sigma=sigma, B=B)
            result.subproblem_time += time() - start_subproblem_time

            if !result.success
                return result
            end
        end

        # Update result
        norm_progress = T(norm(x - result.x, Inf) / max(1, norm(result.x, Inf)))

        # Update lambda
        if !Jf_is_analytic
            scale = options.epsilon / (sigma * sqrt_n)
            lambda_factor_val = options.lambda_factor isa Function ? convert(T, options.lambda_factor(; iter=iter)) : convert(T, options.lambda_factor)

            lambda = ifelse(
                isfinite(scale) && scale > 0,
                max(scale * lambda_factor_val, options.lambda_min),
                options.lambda_min
            )
        end

        JF_prev .= JF
        Jf!(JF, x; lambda=lambda)
        
        # Update B matrices using BFGS
        # BFGS_update!(B, n, m, x, result.x, JF, JF_prev) # TODO: Check here if the B matrices is ill-conditioned
        theta_bfgs = isfinite(theta) ? theta : one(T)  # ensures eps ≈ 1e-6 inside bfgsupdate!
        bfgsupdate!(B, x, result.x, JF, JF_prev, theta_bfgs)

        # Solve the subproblem with multiple iterations
        start_subproblem_time = time()
        theta, p = subproblem(x, JF, h, :pdfpm; sigma=sigma, B=B)
        result.subproblem_time += time() - start_subproblem_time
        if !result.success
            return result
        end
        update!(result, x=x, Fval=Fx, abs_theta=abs(theta), norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)

        if options.verbose > 0
            if result.iter == 0 || (result.iter % options.print_interval == 0)
                print_table(result)
            end
        end

        # Stopping criteria
        # Test optimality
        if options.stop_criteria == :NormProgress
            if norm_progress <= options.ftol && result.abs_theta <= options.opt_tol 
                result.message = "The solution was found using |θ_PDFPM| <= $(options.opt_tol) and norm_progress <= $(options.ftol)"
                result.success = true
                return result
            end
        elseif result.abs_theta <= options.opt_tol
            result.message = "The solution was found using |θ_PDFPM| <= $(options.opt_tol)"
            result.success = true
            return result
        end

        abort = callback(iter, result)

        # External interruption
        if abort === true
            result.success = false   
            result.message = "Interrupted by callback"
            return result
        end
    end
    
    if options.verbose > 0 && result.iter >= options.max_iter
        result.message = "Maximum number of iterations reached $(options.max_iter)"
        @info result.message
        result.success = false
    end
    
    return result
end
