"""
    PDFPMOptions

Public configuration object for the Partially Derivative-Free Proximal Method
(PDFPM) solver.

`PDFPMOptions` does not choose the solver's floating-point type. Numeric solver
options are converted once at the solver boundary to match the element type of
`x0`. The execution-time limit is measured in seconds and converted separately
to `Float64`.
"""
Base.@kwdef struct PDFPMOptions
    verbose::Int = 1
    max_iter::Int = 1000
    opt_tol::Real = 1e-4
    ftol::Real = 1e-4
    max_time::Real = 3600.0
    print_interval::Int = 1
    stop_criteria::Symbol = :pdfpm
    store_trace::Bool = false
    epsilon::Real = 1e-10
    sigma::Real = 1.0
    alpha::Real = 0.1
    max_subproblem_iter::Int = 200
    lambda_factor::Union{Function,Real} = 0.1
    lambda_min::Real = 1e-12
    step_tolerance_rule::Symbol = :epsilon

    function PDFPMOptions(
        verbose,
        max_iter,
        opt_tol,
        ftol,
        max_time,
        print_interval,
        stop_criteria,
        store_trace,
        epsilon,
        sigma,
        alpha,
        max_subproblem_iter,
        lambda_factor,
        lambda_min,
        step_tolerance_rule,
    )
        options = new(
            Int(verbose),
            Int(max_iter),
            opt_tol,
            ftol,
            max_time,
            Int(print_interval),
            stop_criteria,
            Bool(store_trace),
            epsilon,
            sigma,
            alpha,
            Int(max_subproblem_iter),
            lambda_factor,
            lambda_min,
            step_tolerance_rule,
        )
        return _validate_pdfpm_options(options)
    end
end

"""
    struct PDFPMParams{T<:AbstractFloat}

Parameters for the Partially Derivative-Free Proximal Method (PDFPM) solver.

# Fields
- `verbose::Int`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Int`: Maximum number of iterations.
- `opt_tol::T`: Optimality tolerance (convergence criterion).
- `ftol::T`: Function tolerance (convergence criterion).
- `max_time::Float64`: Maximum execution time in seconds.
- `print_interval::Int`: Interval for printing progress (in iterations).
- `stop_criteria::Symbol`: Additional stopping criteria (`:pdfpm` or `:NormProgress`).
- `store_trace::Bool`: Whether to store the trace of iterates `x` during optimization.
- `epsilon::T`: Tolerance parameter ε for the subproblem.
- `sigma::T`: Regularization parameter σ.
- `alpha::T`: Parameter α for sufficient decrease condition.
- `max_subproblem_iter::Int`: Maximum number of iterations for the subproblem.
- `lambda_factor::Union{Function, T}`: Factor to determine the finite difference parameters.
- `lambda_min::T`: Minimum value for the finite difference parameter λ.
- `step_tolerance_rule::Symbol`: Proximal-step and sufficient-decrease thresholds (`:epsilon` or `:trace_scaled`).
"""
struct PDFPMParams{T<:AbstractFloat}
    verbose::Int # Verbose level
    max_iter::Int # Maximum number of iterations
    opt_tol::T # Optimality tolerance
    ftol::T # Function tolerance
    max_time::Float64 # Maximum time in seconds
    print_interval::Int # Print interval
    stop_criteria::Symbol # Additional stop criteria
    store_trace::Bool # Store trace
    epsilon::T # Epsilon parameter for finite differences
    sigma::T # Sigma parameter for regularization
    alpha::T # Alpha parameter for sufficient decrease
    max_subproblem_iter::Int # Maximum subproblem iterations
    lambda_factor::Union{Function, T} # Percentage in [0,1] of the interval [0, ϵ/(σ√n)] used as λ via λ = max(lambda_factor * ϵ/(σ√n), λ_min). If a Function, it is evaluated at runtime; called as lambda_factor(; iter); the return value is converted to T
    lambda_min::T # Minimum lambda for finite differences
    step_tolerance_rule::Symbol # Thresholds of the proximal-step and sufficient-decrease tests
end


"""
    PDFPM_options(; kw...)

Create a public `PDFPMOptions` configuration object.

Numeric solver options are converted to the element type of `x0` at the
`PDFPM` solver boundary. `max_time` is measured in seconds and stored
internally as a `Float64`.

# Keywords
- `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed).
- `max_iter::Integer = 1000`: Maximum number of iterations.
- `opt_tol::Real = 1e-4`: Optimality tolerance.
- `ftol::Real = 1e-4`: Function tolerance.
- `max_time::Real = 3600.0`: Maximum time in seconds.
- `print_interval::Integer = 1`: Print interval.
- `stop_criteria::Symbol = :pdfpm`: Additional stopping criteria (`:pdfpm` or `:NormProgress`).
- `store_trace::Bool = false`: Whether to store the trace of `x` during optimization.
- `epsilon::Real = 1e-10`: Tolerance parameter ε.
- `sigma::Real = 1.0`: Regularization parameter σ.
- `alpha::Real = 0.1`: Parameter α for sufficient decrease.
- `max_subproblem_iter::Integer = 200`: Maximum subproblem iterations.
- `lambda_factor::Union{Real,Function} = 0.1`: Factor for the finite-difference step. Numeric values are clamped to `[0, 1]`; functions are called as `lambda_factor(; iter)`.
- `lambda_min::Real = 1e-12`: Minimum finite difference step λ.
- `step_tolerance_rule::Symbol = :epsilon`: Thresholds of the proximal-step and sufficient-decrease tests.
  With `T = maxⱼ tr(Bⱼ)` for the curvature matrices of the current subproblem:
  - `:epsilon`: stop if `σ‖p-x‖ ≤ ε`; accept `p` if `Fⱼ(p) ≤ Fⱼ(x) - αε²/(2σ)` for all `j`.
  - `:trace_scaled`: stop if `σ‖p-x‖ < ε/(4(1+T))`; accept `p` if `Fⱼ(p) ≤ Fⱼ(x) - αε²/(32σ(1+T)²)` for all `j`.

# Returns
- `PDFPMOptions`: A public configuration object for the PDFPM solver.
"""
function PDFPM_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=1, stop_criteria=:pdfpm, store_trace=false, epsilon=1e-10, sigma=1.0, alpha=0.1, max_subproblem_iter=200, lambda_factor=0.1, lambda_min=1e-12, step_tolerance_rule=:epsilon)
    return PDFPMOptions(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_trace, epsilon, sigma, alpha, max_subproblem_iter, lambda_factor, lambda_min, step_tolerance_rule)
end

function _validate_pdfpm_options(options::PDFPMOptions)
    options.max_iter >= 0 || throw(ArgumentError("`max_iter` must be nonnegative."))
    !isnan(options.max_time) && options.max_time >= 0.0 || throw(ArgumentError("`max_time` must be nonnegative or Inf."))
    options.print_interval > 0 || throw(ArgumentError("`print_interval` must be positive."))
    options.max_subproblem_iter > 0 || throw(ArgumentError("`max_subproblem_iter` must be positive."))
    options.stop_criteria in (:pdfpm, :NormProgress) ||
        throw(ArgumentError("The stop criteria is not supported: $(options.stop_criteria)"))
    options.step_tolerance_rule in (:epsilon, :trace_scaled) ||
        throw(ArgumentError("The step tolerance rule is not supported: $(options.step_tolerance_rule)"))
    return options
end

function _convert_pdfpm_options(::Type{T}, options::PDFPMOptions) where {T<:AbstractFloat}
    opt_tol = T(options.opt_tol)
    ftol = T(options.ftol)
    epsilon = T(options.epsilon)
    sigma = T(options.sigma)
    alpha = T(options.alpha)
    lambda_min = T(options.lambda_min)
    lambda_factor = options.lambda_factor isa Function ? options.lambda_factor : clamp(T(options.lambda_factor), zero(T), one(T))

    isfinite(opt_tol) && opt_tol >= zero(T) || throw(ArgumentError("`opt_tol` must be finite and nonnegative."))
    isfinite(ftol) && ftol >= zero(T) || throw(ArgumentError("`ftol` must be finite and nonnegative."))
    isfinite(epsilon) && epsilon > zero(T) || throw(ArgumentError("`epsilon` must be finite and positive."))
    isfinite(sigma) && sigma > zero(T) || throw(ArgumentError("`sigma` must be finite and positive."))
    isfinite(alpha) && alpha > zero(T) || throw(ArgumentError("`alpha` must be finite and positive."))
    isfinite(lambda_min) && lambda_min > zero(T) || throw(ArgumentError("`lambda_min` must be finite and positive."))
    if !(lambda_factor isa Function)
        isfinite(lambda_factor) || throw(ArgumentError("`lambda_factor` must be finite or a function."))
    end

    return PDFPMParams{T}(
        options.verbose,
        options.max_iter,
        opt_tol,
        ftol,
        Float64(options.max_time),
        options.print_interval,
        options.stop_criteria,
        options.store_trace,
        epsilon,
        sigma,
        alpha,
        options.max_subproblem_iter,
        lambda_factor,
        lambda_min,
        options.step_tolerance_rule,
    )
end

"""
    PDFPM(evalf, A, delta, x0, options; evalJf=nothing, lb=nothing, ub=nothing, callback=noop_callback)

Solve a composite multi-objective optimization problem using the Partially Derivative-Free Proximal Method (PDFPM).

The problem is defined as:
    min_{x ∈ Ω} F(x) = f(x) + h(x)

where f: R^n -> R^m is a smooth vector-valued function, and h: R^n -> R^m is a convex function defined as:
    h_j(x) = max{<x,z> | z ∈ Z_j},   Z_j = { z ∈ R^n | -δ e ≤ A_j z ≤ δ e }

This method can use finite differences if the Jacobian `evalJf` is not provided.

# Arguments
- `evalf::Function`: Evaluate the objective function f(x).
- `A::Vector{<:AbstractMatrix}`: A vector of matrices defining the sets Z_j.
- `delta::Real`: The scaling parameter δ for the sets Z_j.
- `x0::AbstractVector`: Initial guess.
- `options::PDFPMOptions`: Options for the solver (see [`PDFPM_options`]).

# Keywords
- `evalJf::Union{Nothing, Function}`: Evaluate the Jacobian Jf(x). If `nothing`, finite differences are used.
- `lb`: Lower-bounds vector. If omitted, variables are unbounded below.
- `ub`: Upper-bounds vector. If omitted, variables are unbounded above.
- `callback::Function`: Called after a nonconverged outer iteration as `callback(iter, result)`. Return exactly `true` to stop.

# Returns
- `result::OptimResult`: Optimization result containing solution `x`, objective values `Fval`, and other statistics.
"""
function PDFPM(evalf::Function, A::AbstractVector{<:AbstractMatrix}, delta::Real,
    x0::AbstractVector{T}, options::PDFPMOptions=PDFPM_options();
    evalJf::Union{Nothing,Function}=nothing,
    lb=nothing,
    ub=nothing,
    callback=noop_callback) where {T<:AbstractFloat}

    start_time = time()
    params = _convert_pdfpm_options(T, options)
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

    h!(h, x) = evalh!(h, x, SubProbParams)

    function subproblem(x, JF, h, method::Symbol; sigma::T=1.0, B::Union{Nothing, Vector{<:AbstractMatrix{T}}}=nothing)
        if method == :condg
            return subproblem_condg(x, JF, h, SubProbParams; verbose=params.verbose)
        elseif method == :proxgrad
            return subproblem_proxgrad(x, JF, h, SubProbParams; sigma=sigma, verbose=params.verbose)
        elseif method == :pdfpm
            return subproblem_pdfpm(x, JF, h, SubProbParams; sigma=sigma, B=B, verbose=params.verbose)
        else
            error("The subproblem method is not supported: $method")
        end
    end

    _PDFPM_optimize!(result, f!, Jf!, Jf_is_analytic, h!, m, subproblem, params, callback; total_start_time=start_time)
    return result
end

"""
    _PDFPM_optimize!(result, f!, Jf!, Jf_is_analytic, h!, m, subproblem, options, callback; initial_B=nothing)

Internal function that implements the PDFPM algorithm. `initial_B` replaces the
identity as the initial curvature matrices; it is copied and is not part of the
public interface.
"""
function _PDFPM_optimize!(result::OptimResult, f!::Function, Jf!::Function, Jf_is_analytic::Bool, h!::Function, m::Int, subproblem::Function, options::PDFPMParams{T}, callback::Function; total_start_time=time(), initial_B=nothing) where {T<:AbstractFloat}

    result.status = :running
    result.message = "Optimization in progress"

    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    JF_prev = similar(JF)
    h = zeros(T, m)
    h_prev = zeros(T, m)
    #Initialize theta
    theta = T(NaN)

    # Initialize B matrices as identity for each objective
    B = initial_B === nothing ? [Matrix{T}(I, n, n) for _ in 1:m] : [Matrix{T}(Bj) for Bj in initial_B]

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
        print_optimizer_header("PDFPM", options, n, m)
    end

    # Compute F and JF
    f!(Fx, x)
    Jf!(JF, x; lambda=lambda)
    success, message = h!(h, x)

    # Check if h! failed
    if !success
        return _finalize_optimization!(result, options, total_start_time, :evaluation_failure,
                                       "evalh! failed: " * message)
    end

    h_prev .= h

    if !all(isfinite, Fx) || !all(isfinite, JF) || !all(isfinite, h)
        return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                       "Nonfinite initial objective or Jacobian")
    end
    result.Fval .= Fx .+ h
    result.F_init .= result.Fval

    # Solve the subproblem
    start_subproblem_time = time()
    theta, p, success, message = subproblem(x, JF, h, :pdfpm; sigma=sigma, B=B)
    result.subproblem_time += time() - start_subproblem_time
    if !success
        return _finalize_optimization!(result, options, total_start_time, :subproblem_failure,
                                       "The subproblem of PDFPM failed: " * message)
    end
    if !isfinite(T(theta)) || !all(isfinite, p)
        return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                       "The subproblem of PDFPM returned nonfinite theta or p (theta=$theta)")
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
        return _finalize_optimization!(result, options, total_start_time, :optimality_tolerance,
                                       "Solution found at initial point with |θ_PDFPM| ≤ $(options.opt_tol)")
    end

    for iter in 1:options.max_iter

        if time() - start_time >= options.max_time
            return _finalize_optimization!(result, options, total_start_time, :max_time,
                                           "Maximum time reached $(options.max_time)")
        end
        # Store previous value of h
        h_prev .= h

        for sub_iter in 1:options.max_subproblem_iter
            # p was computed with the current B, which is fixed inside this loop
            stop_step, decrease = _pdfpm_step_tolerances(options, sigma, B)
            # Check convergence
            if stop_step(sigma * norm(p - x))
                x .= p

                # Update result
                norm_progress = T(norm(x - result.x, Inf) / max(1, norm(result.x, Inf)))
                # Compute F in-place: Fx = f(x) + h(x)
                f!(Fx, x)
                success, message = h!(h, x)
                # Check if h! failed in F evaluation
                if !success
                    return _finalize_optimization!(result, options, total_start_time, :evaluation_failure,
                                                   "evalh! failed: " * message)
                end
                Fx .+= h
                if !all(isfinite, Fx)
                    return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                                   "Nonfinite objective at the proximal-step stopping point")
                end
                # The model measure was computed at the preceding point, not at p.
                update!(result, x=x, Fval=Fx, abs_theta=T(NaN), norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)
                if options.verbose > 0
                    if iter == 0 || (iter % options.print_interval == 0)
                        print_table(result)
                    end
                end

                return _finalize_optimization!(result, options, total_start_time, :proximal_step_tolerance,
                                               _pdfpm_proximal_step_message(options))
            end

            # Check sufficient decrease
            f!(Fx, p)
            success, message = h!(h, p)

            # Check if h! failed in F evaluation
            if !success
                return _finalize_optimization!(result, options, total_start_time, :evaluation_failure,
                                               "evalh! failed: " * message)
            end

            Fx .+= h

            if all(isfinite, Fx) && all(Fx .<= result.Fval .- decrease)
                # Update x
                x .= p
                break
            else
                if sub_iter >= options.max_subproblem_iter
                    return _finalize_optimization!(result, options, total_start_time, :max_subproblem_iter,
                                                   "The maximum subiteration limit was reached ($(options.max_subproblem_iter)); sigma=$sigma")
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
                    if !all(isfinite, JF)
                        return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                                       "Nonfinite finite-difference Jacobian")
                    end
                end
            end
            # Check maximum time
            if time() - start_time >= options.max_time
                return _finalize_optimization!(result, options, total_start_time, :max_time,
                                               "Maximum time reached $(options.max_time)")
            end
            # Solve the subproblem
            start_subproblem_time = time()
            theta, p, success, message = subproblem(x, JF, h_prev, :pdfpm; sigma=sigma, B=B)
            result.subproblem_time += time() - start_subproblem_time

            if !success
                return _finalize_optimization!(result, options, total_start_time, :subproblem_failure,
                                               "The subproblem of PDFPM failed: " * message)
            end
            if !isfinite(T(theta)) || !all(isfinite, p)
                return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                               "The subproblem of PDFPM returned nonfinite theta or p (theta=$theta)")
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
        if !all(isfinite, JF)
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "Nonfinite Jacobian after the accepted step")
        end

        # Update B matrices using BFGS
        theta_bfgs = isfinite(theta) ? theta : one(T)  # ensures eps ≈ 1e-6 inside bfgsupdate!
        bfgsupdate!(B, x, result.x, JF, JF_prev, theta_bfgs) # TODO: Check here if the B matrices is ill-conditioned

        # Solve the subproblem with multiple iterations
        start_subproblem_time = time()
        theta, p, success, message = subproblem(x, JF, h, :pdfpm; sigma=sigma, B=B)
        result.subproblem_time += time() - start_subproblem_time
        if !success
            return _finalize_optimization!(result, options, total_start_time, :subproblem_failure,
                                           "The subproblem of PDFPM failed: " * message)
        end
        if !isfinite(T(theta)) || !all(isfinite, p)
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "The subproblem of PDFPM returned nonfinite theta or p (theta=$theta)")
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
                return _finalize_optimization!(result, options, total_start_time, :optimality_and_progress_tolerance,
                                               "The solution was found using |θ_PDFPM| <= $(options.opt_tol) and norm_progress <= $(options.ftol)")
            end
        elseif result.abs_theta <= options.opt_tol
            return _finalize_optimization!(result, options, total_start_time, :optimality_tolerance,
                                           "The solution was found using |θ_PDFPM| <= $(options.opt_tol)")
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

"""
    _pdfpm_step_tolerances(options, sigma, B)

Return the proximal-step stopping test and the sufficient-decrease amount of
`options.step_tolerance_rule` for the regularization `sigma` and the curvature
matrices `B` of the current subproblem.

With `T = maxⱼ tr(Bⱼ)`, `:epsilon` stops on `σ‖p-x‖ ≤ ε` and requires
`αε²/(2σ)`; `:trace_scaled` stops on `σ‖p-x‖ < ε/(4(1+T))` and requires
`αε²/(32σ(1+T)²)`. A step norm equal to the threshold therefore stops under
`:epsilon` and proceeds to the decrease test under `:trace_scaled`.
"""
function _pdfpm_step_tolerances(options::PDFPMParams{T}, sigma::T, B) where {T<:AbstractFloat}
    if options.step_tolerance_rule === :epsilon
        return (≤(options.epsilon)), (options.alpha * options.epsilon^2) / (2 * sigma)
    end
    trace_scale = 1 + maximum(tr, B)
    return (<(options.epsilon / (4 * trace_scale))),
           (options.alpha * options.epsilon^2) / (32 * sigma * trace_scale^2)
end

function _pdfpm_proximal_step_message(options::PDFPMParams)
    if options.step_tolerance_rule === :epsilon
        return "PDFPM proximal step satisfies sigma*norm(p-x) <= epsilon"
    end
    return "PDFPM proximal step satisfies sigma*norm(p-x) < epsilon/(4(1+max_j tr(B_j)))"
end
