mutable struct CondGParams{T<:AbstractFloat}
    verbose::Integer # Verbose level
    max_iter::Integer # Maximum number of iterations
    opt_tol::T # Optimality tolerance
    ftol::T # Function tolerance
    max_time::T # Maximum time in seconds
    print_interval::Integer # Print interval
    stop_criteria::Symbol # Additional stop criteria
    store_trace::Bool # Store trace
end


"""	
    CondG_options(; verbose::Integer = 1, max_iter::Integer = 1000, opt_tol::T = 1e-4, ftol::T = 1e-4, max_time::T = 3600.0, print_interval::Integer = 10, store_trace::Bool = false)

# Arguments:
# - `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed)
# - `max_iter::Integer = 1000`: Maximum number of iterations
# - `opt_tol::T = 1e-4`: Optimality tolerance
# - `ftol::T = 1e-4`: Function tolerance
# - `max_time::T = 3600.0`: Maximum time in seconds
# - `print_interval::Integer = 10`: Print interval
# - `stop_criteria::Symbol = :condg`:  Additional stop criteria (:condg, :NormProgress)
# - `store_trace::Bool = false`: Whether to store the trace of x during optimization
"""
function CondG_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=10, stop_criteria=:condg, store_trace=false)
    if stop_criteria ∉ [:condg, :NormProgress]
        error("The stop criteria is not supported: $stop_criteria")
    end
    return CondGParams(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_trace)
end

"""
    CondG(evalf::Function, evalg::Function, x0::AbstractVector{T}, options::CondGParams=CondG_options();callback=noop_callback) where {T<:AbstractFloat}

# Arguments:
# - `evalf::Function`: Function to evaluate the objective function f(x): R^n -> R^m
# - `evalJf::Function`: Function to evaluate the Jacobian of the objective function Jf(x): R^n -> R^{m×n}

# - `x0::AbstractVector{T}`: Initial guess
# - `options::CondGParams=CondG_options()`: Options for the CondG solver
# - `callback::Function=noop_callback`: Optional hook executed every iteration, with signature `callback(iter, result) -> Bool`. Return `true` to stop; any other value continues. Use for logging or custom stopping rules.
"""


function CondG(evalf::Function, evalJf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, 
    x0::AbstractVector{T}, options::CondGParams=CondG_options(); 
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
            theta, p, success, message = subproblem_proxgrad(x, JF, h, SubProbParams; verbose=options.verbose)
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




    _CondG_optimize!(result, f!, Jf!, h!, m, subproblem, options, callback)
    result.total_time = time() - start_time
    return result
end

"""
    _CondG_optimize!(f!, Jf!, h!, x0, options, callback, result)

Internal function that implements the CondG algorithm.
"""
function _CondG_optimize!(result::OptimResult, f!::Function, Jf!::Function, h!::Function, m::Int, subproblem::Function, options::CondGParams, callback::Function)
    

    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    T = eltype(x)
    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    h  = zeros(T, m)

    # Preallocated buffers for line search evaluations F(y) = f(y) + h(y)
    Fx_ls = zeros(T, m)
    h_ls  = zeros(T, m)
    F_ls  = zeros(T, m)

    # Start timing to check for maximum time
    start_time = time()

    # Define the composite objective function F(x) = f(x) + h(x)
    function F(y; m::Int=m)
        # Use try-catch to capture any error during evaluation
        try
            f!(Fx_ls, y)
            h!(h_ls, y)

            # Check if h! failed
            if !result.success
                return nothing, result.message
            end

            F_ls .= Fx_ls
            F_ls .+= h_ls
            return F_ls, nothing
        catch e
            # Capture any other error that may occur
            result.success = false
            result.message = "Error evaluating F: " * string(e)
            return nothing, result.message
        end
    end
    
    if options.verbose > 0
        @info "----------------------------------------------------------------------"
        @info "Conditional Gradient Method for Composite Multiojective Optimization"
        @info "----------------------------------------------------------------------"
        @info "N of variables=$(n), N of objectives=$(m), Opt tol=$(options.opt_tol), Max iter=$(options.max_iter)"
    end

    # Compute F and JF
    f!(Fx, x)
    Jf!(JF, x)
    h!(h, x)
    
    # Check if h! failed
    if !result.success
        return result
    end

    # Solve the subproblem
    start_subproblem_time = time()
    theta, p = subproblem(x, JF, h, :condg)
    result.subproblem_time += time() - start_subproblem_time
    if !result.success
        return result
    end

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
        result.message = "Solution found at initial point with |θ_CondG| ≤ $(options.opt_tol)"
        return result
    end

    # Direction buffer (reused)
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

        if !line_search_result.success
            result.message = line_search_result.message
            result.success = false
            break
        end

        # Update x in-place
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
        theta, p = subproblem(x, JF, h, :condg)
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
                result.message = "The solution was found using |θ_CondG| <= $(options.opt_tol) and norm_progress <= $(options.ftol)"
                break
            end
        elseif result.abs_theta <= options.opt_tol
            result.message = "The solution was found using |θ_CondG| <= $(options.opt_tol)"
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