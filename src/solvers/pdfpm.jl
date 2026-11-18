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
    max_time::Real = 3600.0
    print_interval::Int = 1
    store_level::Symbol = :none
    epsilon::Real = 1e-10
    sigma::Real = 1.0
    alpha::Real = 0.1
    max_subproblem_iter::Int = 200
    lambda_factor::Union{Function,Real} = 0.1
    lambda_min::Real = 1e-12
    max_cond_A::Real = 1e8
    subproblem_tol::Union{Nothing,Real} = nothing
    decrease_rule::Symbol = :proximal_step
    curvature::Symbol = :bfgs

    function PDFPMOptions(
        verbose,
        max_iter,
        max_time,
        print_interval,
        store_level,
        epsilon,
        sigma,
        alpha,
        max_subproblem_iter,
        lambda_factor,
        lambda_min,
        max_cond_A,
        subproblem_tol,
        decrease_rule,
        curvature,
    )
        options = new(
            Int(verbose),
            Int(max_iter),
            max_time,
            Int(print_interval),
            store_level,
            epsilon,
            sigma,
            alpha,
            Int(max_subproblem_iter),
            lambda_factor,
            lambda_min,
            max_cond_A,
            subproblem_tol,
            Symbol(decrease_rule),
            Symbol(curvature),
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
- `max_time::Float64`: Maximum execution time in seconds.
- `print_interval::Int`: Interval for printing progress (in iterations).
- `store_level::Symbol`: Traces kept in the result (`:none`, `:low`, `:medium` or `:high`).
- `epsilon::T`: Tolerance parameter ε for the subproblem.
- `sigma::T`: Regularization parameter σ.
- `alpha::T`: Parameter α for sufficient decrease condition.
- `max_subproblem_iter::Int`: Maximum number of iterations for the subproblem.
- `lambda_factor::Union{Function, T}`: Factor to determine the finite difference parameters.
- `lambda_min::T`: Minimum value for the finite difference parameter λ.
- `max_cond_A::T`: Largest accepted condition number of the matrices `Ãⱼ` when `delta > 0`.
- `subproblem_tol::T`: Tolerance of the subproblem solver.
- `decrease_rule::Symbol`: Decrease required in Step 4 (`:proximal_step` or `:epsilon_trace`).
- `curvature::Symbol`: Curvature matrices `Bⱼ` (`:bfgs` or `:zero`).
"""
struct PDFPMParams{T<:AbstractFloat}
    verbose::Int # Verbose level
    max_iter::Int # Maximum number of iterations
    max_time::Float64 # Maximum time in seconds
    print_interval::Int # Print interval
    store_level::Symbol # Trace level
    epsilon::T # Epsilon parameter for finite differences
    sigma::T # Sigma parameter for regularization
    alpha::T # Alpha parameter for sufficient decrease
    max_subproblem_iter::Int # Maximum subproblem iterations
    lambda_factor::Union{Function, T} # Percentage in [0,1] of the interval [0, ϵ/(σ√n)] used as λ via λ = min(max(lambda_factor * ϵ/(σ√n), λ_min), ϵ/(σ√n)). If a Function, it is evaluated at runtime; called as lambda_factor(; iter); the return value is converted to T
    lambda_min::T # Minimum lambda for finite differences
    max_cond_A::T # Largest accepted condition number of the matrices Ãⱼ
    subproblem_tol::T # Tolerance of the subproblem solver
    decrease_rule::Symbol # Decrease required in Step 4
    curvature::Symbol # Curvature matrices B_j
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
- `max_time::Real = 3600.0`: Maximum time in seconds.
- `print_interval::Integer = 1`: Print interval.
- `store_level::Symbol = :none`: Traces kept in the result. `:none` keeps none; `:low` keeps
  `Fval_trace`, `abs_theta_trace`, `norm_progress_trace`, and the σ and `maxⱼ tr(Bⱼ)` of the
  tests behind each point (`sigma_trace`, `max_tr_B_trace`); `:medium` adds `x_trace`;
  `:high` is currently the same as `:medium`.
- `epsilon::Real = 1e-10`: Tolerance parameter ε. With `T = maxⱼ tr(Bⱼ)` for the curvature
  matrices of the current subproblem, PDFPM stops if `σ‖p-x‖ < ε/(3(1+T))` (Step 3); otherwise
  it accepts `p` if `Fⱼ(p) ≤ Fⱼ(x) - d` for all `j` (Step 4), with `d` given by `decrease_rule`.
- `sigma::Real = 1.0`: Regularization parameter σ.
- `alpha::Real = 0.1`: Parameter α for sufficient decrease.
- `max_subproblem_iter::Integer = 200`: Maximum subproblem iterations.
- `lambda_factor::Union{Real,Function} = 0.1`: Factor for the finite-difference step. Numeric values are clamped to `[0, 1]`; functions are called as `lambda_factor(; iter)`.
- `lambda_min::Real = 1e-12`: Minimum finite difference step λ; it never lifts λ above `ε/(σ√n)`, the bound of Step 1.
- `max_cond_A::Real = 1e8`: Largest accepted `κ₂(Ãⱼ)` when `delta > 0`; see [`robust_subproblem_params`](@ref).
- `subproblem_tol::Union{Nothing,Real} = nothing`: Tolerance of the subproblem solver; `nothing` means `sqrt(eps(T))` for the element type `T` of `x0`.
- `decrease_rule::Symbol = :proximal_step`: Decrease `d` required in Step 4. `:proximal_step`
  is `d = (α/2)σ‖p-x‖²`; `:epsilon_trace` is `d = αε²/(27σ(1+T)²)`, the rule before this option.
- `curvature::Symbol = :bfgs`: Curvature matrices `Bⱼ`. `:bfgs` starts from the identity and
  applies the safeguarded BFGS update after each accepted step; `:zero` keeps `Bⱼ = 0` in every
  iteration, without update, so `T = 0` and Step 3 uses `ε/3`.

# Returns
- `PDFPMOptions`: A public configuration object for the PDFPM solver.
"""
function PDFPM_options(; verbose=1, max_iter=1000, max_time=3600.0, print_interval=1, store_level=:none, epsilon=1e-10, sigma=1.0, alpha=0.1, max_subproblem_iter=200, lambda_factor=0.1, lambda_min=1e-12, max_cond_A=1e8, subproblem_tol=nothing, decrease_rule=:proximal_step, curvature=:bfgs)
    return PDFPMOptions(verbose, max_iter, max_time, print_interval, store_level, epsilon, sigma, alpha, max_subproblem_iter, lambda_factor, lambda_min, max_cond_A, subproblem_tol, decrease_rule, curvature)
end

function _validate_pdfpm_options(options::PDFPMOptions)
    options.max_iter >= 0 || throw(ArgumentError("`max_iter` must be nonnegative."))
    !isnan(options.max_time) && options.max_time >= 0.0 || throw(ArgumentError("`max_time` must be nonnegative or Inf."))
    options.print_interval > 0 || throw(ArgumentError("`print_interval` must be positive."))
    options.max_subproblem_iter > 0 || throw(ArgumentError("`max_subproblem_iter` must be positive."))
    options.store_level in STORE_LEVELS ||
        throw(ArgumentError("The store level is not supported: $(options.store_level)"))
    _check_max_cond_A(options.max_cond_A)
    _check_subproblem_tol(options.subproblem_tol)
    options.decrease_rule in PDFPM_DECREASE_RULES ||
        throw(ArgumentError("The decrease rule is not supported: $(options.decrease_rule); use one of $(PDFPM_DECREASE_RULES)"))
    options.curvature in PDFPM_CURVATURES ||
        throw(ArgumentError("The curvature is not supported: $(options.curvature); use one of $(PDFPM_CURVATURES)"))
    return options
end

"""Decrease rules of Step 4 accepted by `PDFPM_options` (`decrease_rule`)."""
const PDFPM_DECREASE_RULES = (:proximal_step, :epsilon_trace)

"""Curvature matrices accepted by `PDFPM_options` (`curvature`)."""
const PDFPM_CURVATURES = (:bfgs, :zero)

function _convert_pdfpm_options(::Type{T}, options::PDFPMOptions) where {T<:AbstractFloat}
    epsilon = T(options.epsilon)
    sigma = T(options.sigma)
    alpha = T(options.alpha)
    lambda_min = T(options.lambda_min)
    lambda_factor = options.lambda_factor isa Function ? options.lambda_factor : clamp(T(options.lambda_factor), zero(T), one(T))

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
        Float64(options.max_time),
        options.print_interval,
        options.store_level,
        epsilon,
        sigma,
        alpha,
        options.max_subproblem_iter,
        lambda_factor,
        lambda_min,
        T(options.max_cond_A),
        options.subproblem_tol === nothing ? sqrt(eps(T)) : T(options.subproblem_tol),
        options.decrease_rule,
        options.curvature,
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
    SubProbParams = robust_subproblem_params(A, delta_T, lb_vec, ub_vec; max_cond_A=params.max_cond_A)

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
            return subproblem_condg(x, JF, h, SubProbParams; tol=params.subproblem_tol, verbose=params.verbose)
        elseif method == :proxgrad
            return subproblem_proxgrad(x, JF, h, SubProbParams; sigma=sigma, tol=params.subproblem_tol, verbose=params.verbose)
        elseif method == :pdfpm
            return subproblem_pdfpm(x, JF, h, SubProbParams; sigma=sigma, B=B, tol=params.subproblem_tol, verbose=params.verbose)
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
initial curvature matrices of `options.curvature` (identity for `:bfgs`, zero for `:zero`);
it is copied and is not part of the public interface.
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

    # Initial curvature matrices: identity for BFGS, zero for :zero. Zero matrices, not
    # `B = nothing`, which the subproblem reads as the identity.
    B = initial_B !== nothing ? [Matrix{T}(Bj) for Bj in initial_B] :
        options.curvature === :zero ? [zeros(T, n, n) for _ in 1:m] : [Matrix{T}(I, n, n) for _ in 1:m]

    # Initialize sigma
    sigma = options.sigma

    # Initialize lambda (keep computations in type T)
    sqrt_n = sqrt(T(n))
    scale = options.epsilon / (options.sigma * sqrt_n)
    lambda_factor_val = options.lambda_factor isa Function ? convert(T, options.lambda_factor(; iter=0)) : convert(T, options.lambda_factor)

    lambda = ifelse(
        isfinite(scale) && scale > 0,
        min(max(scale * lambda_factor_val, options.lambda_min), scale),
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

    # Update result; no step has been tested yet, so sigma and max_j tr(B_j) are NaN
    update!(result, x=x, abs_theta=abs(theta), Fval=Fx + h, norm_progress=T(NaN),
            sigma=T(NaN), max_tr_B=T(NaN), store_level=options.store_level)
    result.F_init .= result.Fval

    if options.verbose > 0
        if result.iter == 0 || (result.iter % options.print_interval == 0)
            print_table(result)
        end
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
            step_norm = norm(p - x)
            step_threshold, decrease, max_tr_B = _pdfpm_step_tolerances(options, sigma, B, step_norm)
            step_measure = sigma * step_norm
            # Diagnostics of this test; they do not enter the method
            result.sigma = sigma
            result.max_tr_B = max_tr_B
            result.step_measure = step_measure
            result.step_threshold = step_threshold
            result.required_decrease = decrease
            result.actual_decrease = T(NaN)
            # Step 3: the only convergence test
            if step_measure < step_threshold
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
                update!(result, x=x, Fval=Fx, abs_theta=T(NaN), norm_progress=norm_progress, iter=iter,
                        sigma=sigma, max_tr_B=_pdfpm_traced_max_tr(options, B), store_level=options.store_level)
                if options.verbose > 0
                    if iter == 0 || (iter % options.print_interval == 0)
                        print_table(result)
                    end
                end

                return _finalize_optimization!(result, options, total_start_time, :proximal_step_tolerance,
                                               "PDFPM proximal step satisfies sigma*norm(p-x) < epsilon/(3(1+max_j tr(B_j)))")
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
            result.actual_decrease = minimum(result.Fval .- Fx)

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
                result.n_sigma_doublings += 1
                if !Jf_is_analytic
                    # Update lambda (keep computations in type T)
                    scale = options.epsilon / (sigma * sqrt_n)
                    lambda_factor_val = options.lambda_factor isa Function ? convert(T, options.lambda_factor(; iter=iter)) : convert(T, options.lambda_factor)
                    lambda = ifelse(
                        isfinite(scale) && scale > 0,
                        min(max(scale * lambda_factor_val, options.lambda_min), scale),
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
                min(max(scale * lambda_factor_val, options.lambda_min), scale),
                options.lambda_min
            )
        end

        JF_prev .= JF
        Jf!(JF, x; lambda=lambda)
        if !all(isfinite, JF)
            return _finalize_optimization!(result, options, total_start_time, :numerical_failure,
                                           "Nonfinite Jacobian after the accepted step")
        end

        # max_j tr(B_j) of the subproblem that produced the accepted step, before the update
        step_max_tr_B = _pdfpm_traced_max_tr(options, B)

        # Update B matrices using BFGS; with :zero they stay zero
        if options.curvature === :bfgs
            theta_bfgs = isfinite(theta) ? theta : one(T)  # ensures eps ≈ 1e-6 inside bfgsupdate!
            bfgsupdate!(B, x, result.x, JF, JF_prev, theta_bfgs) # TODO: Check here if the B matrices is ill-conditioned
        end

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
        update!(result, x=x, Fval=Fx, abs_theta=abs(theta), norm_progress=norm_progress, iter=iter,
                sigma=sigma, max_tr_B=step_max_tr_B, store_level=options.store_level)

        if options.verbose > 0
            if result.iter == 0 || (result.iter % options.print_interval == 0)
                print_table(result)
            end
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
    _pdfpm_step_tolerances(options, sigma, B, step_norm)

Return the thresholds of Steps 3 and 4 of the method for the regularization `sigma`, the
curvature matrices `B` of the current subproblem and the step norm `‖p-x‖`, followed by
`T = maxⱼ tr(Bⱼ)`: the proximal step stops on `σ‖p-x‖ < ε/(3(1+T))`, and acceptance requires
the decrease `(α/2)σ‖p-x‖²` (`decrease_rule = :proximal_step`) or `αε²/(27σ(1+T)²)`
(`:epsilon_trace`).
"""
function _pdfpm_step_tolerances(options::PDFPMParams{T}, sigma::T, B, step_norm::T) where {T<:AbstractFloat}
    max_tr_B = maximum(tr, B)
    trace_scale = 1 + max_tr_B
    decrease = options.decrease_rule === :proximal_step ? options.alpha / 2 * sigma * step_norm^2 :
               (options.alpha * options.epsilon^2) / (27 * sigma * trace_scale^2)
    return options.epsilon / (3 * trace_scale), decrease, max_tr_B
end

# maxⱼ tr(Bⱼ) for the trace, computed only when it is recorded.
_pdfpm_traced_max_tr(options::PDFPMParams, B) = options.store_level === :none ? nothing : maximum(tr, B)
