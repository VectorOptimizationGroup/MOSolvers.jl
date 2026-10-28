"""
    CondGOptions

Public configuration object for the Conditional Gradient (CondG) solver.

`CondGOptions` does not choose the solver's floating-point type. Numeric solver
options are converted once at the solver boundary to match the element type of
`x0`. The execution-time limit is measured in seconds and converted separately
to `Float64`.
"""
Base.@kwdef struct CondGOptions
    verbose::Int = 1
    max_iter::Int = 1000
    opt_tol::Real = 1e-4
    ftol::Real = 1e-4
    max_time::Real = 3600.0
    print_interval::Int = 1
    stop_criteria::Symbol = :condg
    store_level::Symbol = :none

    function CondGOptions(
        verbose,
        max_iter,
        opt_tol,
        ftol,
        max_time,
        print_interval,
        stop_criteria,
        store_level,
    )
        options = new(
            Int(verbose),
            Int(max_iter),
            opt_tol,
            ftol,
            max_time,
            Int(print_interval),
            stop_criteria,
            store_level,
        )
        return _validate_condg_options(options)
    end
end

"""
    struct CondGParams{T<:AbstractFloat}

Parameters for the Conditional Gradient (CondG) solver.

# Fields
- `verbose::Int`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Int`: Maximum number of iterations.
- `opt_tol::T`: Optimality tolerance (convergence criterion).
- `ftol::T`: Function tolerance (convergence criterion).
- `max_time::Float64`: Maximum execution time in seconds.
- `print_interval::Int`: Interval for printing progress (in iterations).
- `stop_criteria::Symbol`: Additional stopping criteria (`:condg` or `:NormProgress`).
- `store_level::Symbol`: Traces kept in the result (`:none`, `:low`, `:medium` or `:high`).
"""
struct CondGParams{T<:AbstractFloat}
    verbose::Int # Verbose level
    max_iter::Int # Maximum number of iterations
    opt_tol::T # Optimality tolerance
    ftol::T # Function tolerance
    max_time::Float64 # Maximum time in seconds
    print_interval::Int # Print interval
    stop_criteria::Symbol # Additional stop criteria
    store_level::Symbol # Trace level
end


"""
    CondG_options(; kw...)

Create a public `CondGOptions` configuration object.

Numeric tolerances are converted to the element type of `x0` at the `CondG`
solver boundary. `max_time` is measured in seconds and stored internally as a
`Float64`.

# Keywords
- `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Integer = 1000`: Maximum number of iterations.
- `opt_tol::Real = 1e-4`: Optimality tolerance.
- `ftol::Real = 1e-4`: Function tolerance.
- `max_time::Real = 3600.0`: Maximum time in seconds.
- `print_interval::Integer = 1`: Print interval.
- `stop_criteria::Symbol = :condg`: Additional stopping criteria (`:condg` or `:NormProgress`).
- `store_level::Symbol = :none`: Traces kept in the result. `:none` keeps none; `:low` keeps
  `Fval_trace`, `abs_theta_trace` and `norm_progress_trace`; `:medium` adds
  `x_trace`; `:high` is currently the same as `:medium`.

# Returns
- `CondGOptions`: A public configuration object for the CondG solver.
"""
function CondG_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=1, stop_criteria=:condg, store_level=:none)
    return CondGOptions(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_level)
end

function _validate_condg_options(options::CondGOptions)
    options.max_iter >= 0 || throw(ArgumentError("`max_iter` must be nonnegative."))
    !isnan(options.max_time) && options.max_time >= 0.0 || throw(ArgumentError("`max_time` must be nonnegative or Inf."))
    options.print_interval > 0 || throw(ArgumentError("`print_interval` must be positive."))
    options.stop_criteria in (:condg, :NormProgress) ||
        throw(ArgumentError("The stop criteria is not supported: $(options.stop_criteria)"))
    options.store_level in STORE_LEVELS ||
        throw(ArgumentError("The store level is not supported: $(options.store_level)"))
    return options
end

function _convert_condg_options(::Type{T}, options::CondGOptions) where {T<:AbstractFloat}
    opt_tol = T(options.opt_tol)
    ftol = T(options.ftol)

    isfinite(opt_tol) && opt_tol >= zero(T) || throw(ArgumentError("`opt_tol` must be finite and nonnegative."))
    isfinite(ftol) && ftol >= zero(T) || throw(ArgumentError("`ftol` must be finite and nonnegative."))

    return CondGParams{T}(
        options.verbose,
        options.max_iter,
        opt_tol,
        ftol,
        Float64(options.max_time),
        options.print_interval,
        options.stop_criteria,
        options.store_level,
    )
end

"""
    CondG(evalf, evalJf, A, delta, x0, options; lb=nothing, ub=nothing, callback=noop_callback,
          linesearch=BackTracking())

Solve a composite multi-objective optimization problem using the Conditional Gradient (CondG) method.

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
- `options::CondGOptions`: Options for the solver (see [`CondG_options`]).

# Keywords
- `lb`: Lower-bounds vector. If omitted, variables are unbounded below.
- `ub`: Upper-bounds vector. If omitted, variables are unbounded above.
- `callback::Function`: Called after a nonconverged outer iteration as `callback(iter, result)`. Return exactly `true` to stop.
- `linesearch = BackTracking()`: Line search along `d = p - x`, from
  MultiObjectiveLineSearches, configured once in the element type of `x0`.
  Steps must not exceed 1: a larger `maxstep` or initial step raises
  `ArgumentError` before any evaluation.

# Returns
- `result::OptimResult`: Optimization result containing solution `x`, objective values `Fval`, and other statistics.
"""
function CondG(evalf::Function, evalJf::Function, A::AbstractVector{<:AbstractMatrix}, delta::Real,
    x0::AbstractVector{T}, options::CondGOptions=CondG_options();
    lb=nothing,
    ub=nothing,
    callback=noop_callback,
    linesearch=BackTracking()) where {T<:AbstractFloat}

    start_time = time()
    params = _convert_condg_options(T, options)
    x0_vec = Vector{T}(x0)
    delta_T = T(delta)
    isfinite(delta_T) && delta_T >= zero(T) || throw(ArgumentError("`delta` must be finite and nonnegative."))

    n = length(x0_vec)
    m = length(A)

    # Validations
    @assert m > 0 "At least one matrix A must be provided"

    for i in 1:m
        @assert size(A[i], 1) == n && size(A[i], 2) == n "Matrix A[$i] must be square of size $n×$n"
    end

    A_T = [T.(A[i]) for i in 1:m]
    A2n = [[A_T[i]; -A_T[i]] for i in 1:m]
    b2n = [delta_T * ones(T, 2*n) for i in 1:m]

    # Process bounds
    lb_vec = _convert_solver_bounds(T, lb, n, T(-Inf), :lb)
    ub_vec = _convert_solver_bounds(T, ub, n, T(Inf), :ub)

    # Check if lb <= ub
    for i in 1:n
        @assert lb_vec[i] <= ub_vec[i] "Lower bound lb[$i] = $(lb_vec[i]) must be less than or equal to upper bound ub[$i] = $(ub_vec[i])"
    end

    # Check if x0 is within bounds
    for i in 1:n
        if !(lb_vec[i] <= x0_vec[i] <= ub_vec[i])
            @warn "x0[$i] = $(x0_vec[i]) is out of bounds [$(lb_vec[i]), $(ub_vec[i])]"
        end
    end

    # Create the structure internally
    SubProbParams = SubProblemParams(A2n, b2n, lb_vec, ub_vec)

    # Create result structure
    result = OptimResult(x0_vec, fill(T(NaN), m))

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

    h!(h, x) = evalh!(h, x, SubProbParams)

    function subproblem(x, JF, h, method::Symbol)
        if method == :condg
            return subproblem_condg(x, JF, h, SubProbParams; verbose=params.verbose)
        elseif method == :proxgrad
            return subproblem_proxgrad(x, JF, h, SubProbParams; verbose=params.verbose)
        elseif method == :pdfpm
            return subproblem_pdfpm(x, JF, h, SubProbParams; verbose=params.verbose)
        else
            error("The subproblem method is not supported: $method")
        end
    end

    _CondG_optimize!(result, f!, Jf!, h!, m, subproblem, params, callback; linesearch, total_start_time=start_time)
    return result
end

"""
    _CondG_optimize!(result, f!, Jf!, h!, m, subproblem, options, callback; linesearch)

Internal function that implements the CondG algorithm.
"""
function _CondG_optimize!(result::OptimResult, f!::Function, Jf!::Function, h!::Function, m::Int, subproblem::Function, options::CondGParams{T}, callback::Function; linesearch, total_start_time=time()) where {T<:AbstractFloat}
    ls = configure_linesearch(linesearch, T)

    result.status = :running
    result.message = "Optimization in progress"

    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    h  = zeros(T, m)

    # Preallocated buffers for line search evaluations F(y) = f(y) + h(y)
    Fx_ls = zeros(T, m)
    h_ls  = zeros(T, m)

    # Start timing to check for maximum time
    start_time = time()

    # Composite objective F(y) = f(y) + h(y) for the line search: fill Fy and
    # return nothing, or return the failure message
    function F!(Fy, y)
        try
            f!(Fx_ls, y)
            success, message = h!(h_ls, y)
            success || return "evalh! failed: " * message
            Fy .= Fx_ls .+ h_ls
            return nothing
        catch e
            # Preserve the existing policy for evaluator exceptions in the search.
            return "Error evaluating F: " * string(e)
        end
    end

    if options.verbose > 0
        print_optimizer_header("CondG", options, n, m)
    end

    # Compute F and JF
    f!(Fx, x)
    Jf!(JF, x)
    success, message = h!(h, x)

    # Check if h! failed
    if !success
        return _finalize_optimization!(result, options, total_start_time, :evaluation_failure,
                                       "evalh! failed: " * message)
    end

    if !all(isfinite, Fx) || !all(isfinite, JF) || !all(isfinite, h)
        return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                       "Nonfinite initial objective or Jacobian")
    end
    result.Fval .= Fx .+ h
    result.F_init .= result.Fval

    # Solve the subproblem
    start_subproblem_time = time()
    theta, p, success, message = subproblem(x, JF, h, :condg)
    result.subproblem_time += time() - start_subproblem_time
    if !success
        return _finalize_optimization!(result, options, total_start_time, :subproblem_failure,
                                       "The subproblem of CondG failed: " * message)
    end
    if !isfinite(T(theta)) || !all(isfinite, p)
        return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                       "The subproblem of CondG returned nonfinite theta or p (theta=$theta)")
    end

    update!(result, x=x, abs_theta=abs(theta), Fval=Fx + h, norm_progress=T(NaN), store_level=options.store_level)
    result.F_init .= result.Fval

    if options.verbose > 0
        if result.iter == 0 || (result.iter % options.print_interval == 0)
            print_table(result)
        end
    end

    # Stopping criteria
    if result.abs_theta <= options.opt_tol
        return _finalize_optimization!(result, options, total_start_time, :optimality_tolerance,
                                       "Solution found at initial point with |θ_CondG| ≤ $(options.opt_tol)")
    end

    # Direction and line-search buffers (reused)
    d = similar(x)
    x_trial = similar(x)
    F_trial = zeros(T, m)
    slopes = zeros(T, m)

    for iter in 1:options.max_iter
        if time() - start_time >= options.max_time
            return _finalize_optimization!(result, options, total_start_time, :max_time,
                                           "Maximum time reached $(options.max_time)")
        end

        # x is critical when θ = 0; the line search needs a finite, nonzero θ
        if iszero(T(theta))
            return _finalize_optimization!(result, options, total_start_time, :optimality_tolerance,
                                           "The solution was found with |θ_CondG| = 0")
        elseif !isfinite(T(theta))
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "The subproblem of CondG returned θ = $theta")
        end

        # Define the search direction (reuse buffer)
        d .= p .- x

        # Compute the step size
        ls_result = perform_linesearch!(ls, F!, x, JF, d, theta, result.Fval, x_trial, F_trial, slopes)

        # An evaluation failure already carries its message
        if !ls_result.success
            status = ls_result.reason === :evaluation_failure ? :evaluation_failure : :line_search_failure
            message = ls_result.reason === :evaluation_failure ? ls_result.message :
                      "Line search failed: " * ls_result.message
            return _finalize_optimization!(result, options, total_start_time, status, message)
        end

        # Accept the trial point
        x .= x_trial

        norm_progress = T(norm(x - result.x, Inf) / max(1, norm(result.x, Inf)))

        # Compute F and JF
        f!(Fx, x)
        Jf!(JF, x)
        # Reduces the cost of calling h!: h .= F_trial - Fx (in-place)
        h .= F_trial
        h .-= Fx

        if !all(isfinite, Fx) || !all(isfinite, JF) || !all(isfinite, h)
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "Nonfinite objective or Jacobian after the accepted step")
        end

        # Solve the subproblem
        start_subproblem_time = time()
        theta, p, success, message = subproblem(x, JF, h, :condg)
        result.subproblem_time += time() - start_subproblem_time
        if !success
            return _finalize_optimization!(result, options, total_start_time, :subproblem_failure,
                                           "The subproblem of CondG failed: " * message)
        end
        if !isfinite(T(theta)) || !all(isfinite, p)
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "The subproblem of CondG returned nonfinite theta or p (theta=$theta)")
        end

        # Update
        update!(result, x=x, Fval=F_trial, abs_theta=abs(theta), norm_progress=norm_progress, iter=iter, store_level=options.store_level)

        if options.verbose > 0
            if result.iter == 0 || (result.iter % options.print_interval == 0)
                print_table(result)
            end
        end

        # Stopping criteria
        # Test optimality
        if options.stop_criteria == :NormProgress
            if norm_progress <= options.ftol && result.abs_theta <= options.opt_tol
                return _finalize_optimization!(result, options, total_start_time, :optimality_and_progress_tolerance,
                                               "The solution was found using |θ_CondG| <= $(options.opt_tol) and norm_progress <= $(options.ftol)")
            end
        elseif result.abs_theta <= options.opt_tol
            return _finalize_optimization!(result, options, total_start_time, :optimality_tolerance,
                                           "The solution was found using |θ_CondG| <= $(options.opt_tol)")
        end

        abort = callback(iter, result)

        # External interruption
        if abort === true
            return _finalize_optimization!(result, options, total_start_time, :callback,
                                           "Interrupted by callback")
        end
    end

    # Only exhaustion reaches this epilogue; other causes return above.
    return _finalize_optimization!(result, options, total_start_time, :max_iter,
                                   "Maximum number of iterations reached $(options.max_iter)")
end
