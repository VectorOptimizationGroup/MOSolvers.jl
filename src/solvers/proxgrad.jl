"""
    mutable struct ProxGradParams{T<:AbstractFloat}

Parameters for the Proximal Gradient solver.

# Fields
- `verbose::Integer`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Integer`: Maximum number of iterations.
- `opt_tol::T`: Optimality tolerance (convergence criterion).
- `ftol::T`: Function tolerance (convergence criterion).
- `max_time::T`: Maximum execution time in seconds.
- `print_interval::Integer`: Interval for printing progress (in iterations).
- `stop_criteria::Symbol`: Additional stopping criteria (`:proxgrad` or `:NormProgress`).
- `store_trace::Bool`: Whether to store the trace of iterates `x` during optimization.
- `mu::T`: Proximal parameter μ.
"""
mutable struct ProxGradParams{T<:AbstractFloat}
    verbose::Integer # Verbose level
    max_iter::Integer # Maximum number of iterations
    opt_tol::T # Optimality tolerance
    ftol::T # Function tolerance
    max_time::T # Maximum time in seconds
    print_interval::Integer # Print interval
    stop_criteria::Symbol # Additional stop criteria
    store_trace::Bool # Store trace
    mu::T # Proximal parameter
end


"""
    ProxGrad_options(; kw...)

Create a `ProxGradParams` structure with default or specified values.

# Keywords
- `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Integer = 1000`: Maximum number of iterations.
- `opt_tol::T = 1e-4`: Optimality tolerance.
- `ftol::T = 1e-4`: Function tolerance.
- `max_time::T = 3600.0`: Maximum time in seconds.
- `print_interval::Integer = 10`: Print interval.
- `stop_criteria::Symbol = :proxgrad`: Additional stopping criteria (`:proxgrad` or `:NormProgress`).
- `store_trace::Bool = false`: Whether to store the trace of `x` during optimization.
- `mu::T = 1.0`: Proximal parameter μ.

# Returns
- `ProxGradParams`: A parameter structure for the ProxGrad solver.
"""
function ProxGrad_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=10, stop_criteria=:proxgrad, store_trace=false, mu=1.0)
    if stop_criteria ∉ [:NormProgress, :proxgrad]
        error("The stop criteria is not supported: $stop_criteria")
    end
    return ProxGradParams(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_trace, mu)
end

"""
    ProxGrad(evalf, evalJf, A, delta, x0, options; lb=nothing, ub=nothing, callback=noop_callback)

Solve a composite multi-objective optimization problem using the Proximal Gradient method.

The problem is defined as:
    min_{x ∈ Ω} F(x) = f(x) + h(x)

where f: R^n -> R^m is a smooth vector-valued function, and h: R^n -> R^m is a convex function defined as:
    h_j(x) = max{<x,z> | z ∈ Z_j},   Z_j = { z ∈ R^n | -δ e ≤ A_j z ≤ δ e }

# Arguments
- `evalf::Function`: Evaluate the objective function f(x).
- `evalJf::Function`: Evaluate the Jacobian of the objective function Jf(x).
- `A::Vector{<:AbstractMatrix}`: A vector of matrices defining the sets Z_j.
- `delta::Real`: The scaling parameter δ for the sets Z_j.
- `x0::AbstractVector`: Initial guess.
- `options::ProxGradParams`: Options for the solver (see [`ProxGrad_options`]).

# Keywords
- `lb::Union{Nothing, AbstractVector, Number}`: Lower bounds on variables.
- `ub::Union{Nothing, AbstractVector, Number}`: Upper bounds on variables.
- `callback::Function`: Optional callback executed every iteration, signature `callback(iter, result) -> Bool`. Return `true` to stop.

# Returns
- `result::OptimResult`: Optimization result containing solution `x`, objective values `Fval`, and other statistics.
"""

function ProxGrad(evalf::Function, evalJf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, 
    x0::AbstractVector{T}, options::ProxGradParams=ProxGrad_options(); 
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

    function Jf!(Jf, x)
        result.n_Jf_evals += 1
        Jf .= evalJf(x)
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

    function subproblem(x, JF, h, method::Symbol)
        if method == :condg
            theta, p, success, message = subproblem_condg(x, JF, h, SubProbParams; verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of CondG failed: " * message
            end
            return theta, p
        elseif method == :proxgrad
            theta, p, success, message = subproblem_proxgrad(x, JF, h, SubProbParams; sigma=options.mu, verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of ProxGrad failed: " * message
            end
            return theta, p
        elseif method == :pdfpm
            theta, p, success, message = subproblem_pdfpm(x, JF, h, SubProbParams; verbose=options.verbose)
            if !success
                result.success = false
                result.message = "The subproblem of PDFPM failed: " * message
            end
            return theta, p
        else
            error("The subproblem method is not supported: $method")
        end
    end

    _ProxGrad_optimize!(result, f!, Jf!, h!, m, subproblem, options, callback)
    result.total_time = time() - start_time
    return result
end

"""
    _ProxGrad_optimize!(f!, Jf!, h!, x0, options, callback, result)

Internal function that implements the ProxGrad algorithm.
"""
function _ProxGrad_optimize!(result::OptimResult, f!::Function, Jf!::Function, h!::Function, m::Int, subproblem::Function, options::ProxGradParams, callback::Function)
    
    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    T = eltype(x)
    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    h  = zeros(T, m)

    # Preallocated buffers for line-search evaluations F(y) = f(y) + h(y)
    Fx_ls = zeros(T, m)
    h_ls  = zeros(T, m)
    F_ls  = zeros(T, m)

    # Start timing to check for maximum time
    start_time = time()

    # Define the composite objective function F(y) = f(y) + h(y) (in-place)
    function F(y; m::Int=m)
        try
            f!(Fx_ls, y)
            h!(h_ls, y)
            if !result.success
                return nothing, result.message
            end
            F_ls .= Fx_ls
            F_ls .+= h_ls
            return F_ls, nothing
        catch e
            result.success = false
            result.message = "Error evaluating F: " * string(e)
            return nothing, result.message
        end
    end
    
    if options.verbose > 0
        @info "----------------------------------------------------------------------"
        @info "Proximal Gradient Method for Composite Multiojective Optimization"
        @info "----------------------------------------------------------------------"
        @info "N of variables=$(n), N of objectives=$(m), Opt tol=$(options.opt_tol), Max iter=$(options.max_iter), mu=$(options.mu)"
    end

    # Initial evaluation
    f!(Fx, x)
    Jf!(JF, x)
    h!(h, x)

    # Check if h! failed
    if !result.success
        return result
    end
    
    F_ls .= Fx
    F_ls .+= h
    update!(result, x=x, Fval=F_ls, store_trace=options.store_trace)
    result.F_init .= result.Fval

    # Solve the subproblem
    start_subproblem_time = time()
    theta, p = subproblem(x, JF, h, :proxgrad)
    result.subproblem_time += time() - start_subproblem_time
    if !result.success
        return result
    end

    F_ls .= Fx
    F_ls .+= h
    update!(result, x=x, abs_theta=abs(theta), Fval=F_ls, norm_progress=T(NaN), store_trace=options.store_trace)
    result.F_init .= result.Fval

    if options.verbose > 0
        if result.iter == 0 || (result.iter % options.print_interval == 0)
            print_table(result)
        end
    end

    # Stopping criteria
    if result.abs_theta <= options.opt_tol
        result.success = true
        result.message = "Solution found at initial point with |θ_ProxGrad| ≤ $(options.opt_tol)"
        return result
    end

    # Reusable direction buffer
    d = similar(x)
    for iter in 1:options.max_iter
        if time() - start_time >= options.max_time
            result.success = false
            result.message = "Maximum time reached $(options.max_time)"
            return result
        end

        # Define the search direction (reuse buffer)
        d .= p .- x

        # Compute the step size
        line_search_result = armijo(F, x, d, result.Fval, JF, factor=-abs(theta))
        
        # If line search failed due to an error in F, the message has already been updated
        if !line_search_result.success
            result.message = line_search_result.message
            result.success = false
            break
        end

        # Update x (in-place)
        x .+= line_search_result.stp .* d

        norm_progress = T(norm(x - result.x, Inf) / max(1, norm(result.x, Inf)))

        # Compute F and JF
        f!(Fx, x)
        Jf!(JF, x)
        # Reduces the cost of calling h!: h .= F_final - Fx (in-place)
        h .= line_search_result.F_final
        h .-= Fx

        # Solve the subproblem
        start_subproblem_time = time()
        theta, p = subproblem(x, JF, h, :proxgrad)
        result.subproblem_time += time() - start_subproblem_time
        if !result.success
            break
        end

        # Update
        update!(result, x=x, Fval=line_search_result.F_final, abs_theta=abs(theta), norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)

        if options.verbose > 0
            if result.iter == 0 || (result.iter % options.print_interval == 0)
                print_table(result)
            end
        end

        # Stopping criteria
        # Test optimality
        if options.stop_criteria == :NormProgress
            if norm_progress <= options.ftol && result.abs_theta <= options.opt_tol 
                result.message = "The solution was found using |θ_ProxGrad| <= $(options.opt_tol) and norm_progress <= $(options.ftol)"
                break
            end
        elseif result.abs_theta <= options.opt_tol
            result.message = "The solution was found using |θ_ProxGrad| <= $(options.opt_tol)"
            break
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
