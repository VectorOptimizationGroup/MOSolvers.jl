mutable struct DFreeMOParams{T<:AbstractFloat}
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
    lambda_min::T # Minimum lambda for finite differences
end


"""	
    DFreeMO_options(; verbose::Integer = 1, max_iter::Integer = 1000, opt_tol::T = 1e-4, ftol::T = 1e-4, max_time::T = 3600.0, print_interval::Integer = 10, store_trace::Bool = false, epsilon::T = 1e-10, sigma::T = 1.0, alpha::T = 0.1, max_subproblem_iter::Integer = 200)

# Arguments:
# - `verbose::Integer = 1`: Verbose level (0: none, 1: normal, 2: detailed)
# - `max_iter::Integer = 1000`: Maximum number of iterations
# - `opt_tol::T = 1e-4`: Optimality tolerance
# - `ftol::T = 1e-4`: Function tolerance
# - `max_time::T = 3600.0`: Maximum time in seconds
# - `print_interval::Integer = 10`: Print interval
# - `stop_criteria::Symbol = :dfreemo`:  Additional stop criteria (:condg, :proxgrad, :dfreemo)
# - `store_trace::Bool = false`: Whether to store the trace of x during optimization
# - `epsilon::T = 1e-10`: Epsilon parameter for finite differences
# - `sigma::T = 1.0`: Sigma parameter for regularization
# - `alpha::T = 0.1`: Alpha parameter for sufficient decrease
# - `max_subproblem_iter::Integer = 200`: Maximum subproblem iterations
"""
function DFreeMO_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=10, stop_criteria=:dfreemo, store_trace=false, epsilon=1e-10, sigma=1.0, alpha=0.1, max_subproblem_iter=200, lambda_min=1e-12)
    if stop_criteria ∉ [:condg, :proxgrad, :dfreemo]
        error("The stop criteria is not supported: $stop_criteria")
    end
    return DFreeMOParams(verbose, max_iter, opt_tol, ftol, max_time, print_interval, stop_criteria, store_trace, epsilon, sigma, alpha, max_subproblem_iter, lambda_min)
end

"""
    DFreeMO(evalf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, x0::AbstractVector{T}, options::DFreeMOParams=DFreeMO_options(); evalJf::Union{Nothing,Function}=nothing, lb::Union{Nothing,AbstractVector{T},Number}=nothing, ub::Union{Nothing,AbstractVector{T},Number}=nothing, callback=noop_callback) where {T<:AbstractFloat}

# Arguments:
# - `evalf::Function`: Function to evaluate the objective function f(x): R^n -> R^m
# - `A::Vector{<:AbstractMatrix{T}}`: Vector of matrices A for each objective
# - `delta::T`: Uncertainty parameter
# - `x0::AbstractVector{T}`: Initial guess
# - `options::DFreeMOParams=DFreeMO_options()`: Options for the DFreeMO solver
# - `evalJf::Union{Nothing,Function}=nothing`: Function to evaluate the Jacobian of the objective function Jf(x): R^n -> R^{m×n} (optional, uses finite differences if not provided)
# - `lb::Union{Nothing,AbstractVector{T},Number}=nothing`: Lower bounds
# - `ub::Union{Nothing,AbstractVector{T},Number}=nothing`: Upper bounds
# - `callback::Function=noop_callback`: Optional hook executed every iteration, with signature `callback(iter, result) -> Bool`. Return `true` to stop; any other value continues. Use for logging or custom stopping rules.
"""

function DFreeMO(evalf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, 
    x0::AbstractVector{T}, options::DFreeMOParams=DFreeMO_options(); 
    evalJf::Union{Nothing,Function}=nothing,
    lb::Union{Nothing,AbstractVector{T},Number}=nothing, 
    ub::Union{Nothing,AbstractVector{T},Number}=nothing,
    callback=noop_callback) where {T<:AbstractFloat}

    start_time = time()

    n = length(x0)
    m = length(A)
    A2n = [[A[i]; -A[i]] for i in 1:m]
    b2n = [delta * ones(2*n) for i in 1:m]

    # Validações
    @assert m > 0 "Pelo menos uma matriz A deve ser fornecida"

    for i in 1:m
        @assert size(A[i], 1) == n && size(A[i], 2) == n "A matriz A[$i] deve ser quadrada de tamanho $n×$n"
    end

    # Processamento dos limites
    lb_vec = process_bounds(lb, n, T(-Inf))
    ub_vec = process_bounds(ub, n, T(Inf))

    # Verificar se lb <= ub
    for i in 1:n
        @assert lb_vec[i] <= ub_vec[i] "O limite inferior lb[$i] = $(lb_vec[i]) deve ser menor ou igual ao limite superior ub[$i] = $(ub_vec[i])"
    end

    # Verificar se x0 está dentro dos limites
    for i in 1:n
        if !(lb_vec[i] <= x0[i] <= ub_vec[i])
            @warn "x0[$i] = $(x0[i]) está fora dos limites [$(lb_vec[i]), $(ub_vec[i])]"
        end
    end

    # Criar a estrutura internamente
    SubProbParams = SubProblemParams(A2n, b2n, lb_vec, ub_vec)

    # Create result structure
    Fval = zeros(T, m)
    result = OptimResult(x0, Fval)

    function f!(f, x)
        result.n_f_evals += 1
        f .= evalf(x)
        return f
    end

    # Criar função Jf! baseada em se evalJf foi fornecido ou não
    function Jf!(Jf, x; method=:central, lambda=sqrt(eps(T)))
        result.n_Jf_evals += 1
        if evalJf !== nothing
            Jf .= evalJf(x)
        else
            # Usar diferenças finitas
            Jf .= jacobian_finite_diff(f!, x, m, method=method, λ=lambda)
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
            theta, p, success, message = subproblem_condg(x, JF, h, SubProbParams)
            if !success
                result.success = false
                result.message = "The subproblem of CondG failed: " * message
            elseif abs(theta) <= 1e-15
                result.success = true
                result.message = "The solution was found"
            end
            return theta, p
        elseif method == :proxgrad
            theta, p, success, message = subproblem_proxgrad(x, JF, h, SubProbParams; sigma=sigma)
            if !success
                result.success = false
                result.message = "The subproblem of ProxGrad failed: " * message
            elseif abs(theta) <= 1e-15
                result.success = true
                result.message = "The solution was found"
            end
            return theta, p
        elseif method == :dfreemo
            theta, p, success, message = subproblem_dfreemo(x, JF, h, SubProbParams; sigma=sigma, B=B)
            # TODO: MOVER ESSA MUDANCA DE SUCCESS DIRETAMENTE PARA O SOLVER
            if !success
                result.success = false
                result.message = "The subproblem of DFreeMO failed: " * message
            elseif abs(theta) <= 1e-15
                result.success = true # QUESTION: If the struct result is initialized with success = true, why is it necessary to set it to true again?
                result.message = "The solution was found"
            end
            return theta, p
        else
            error("The subproblem method is not supported: $method")
        end
    end

    _DFreeMO_optimize!(result, f!, Jf!, h!, m, subproblem, options, callback)
    result.total_time = time() - start_time
    return result
end

"""
    _DFreeMO_optimize!(f!, Jf!, h!, x0, options, callback, result)

Função interna que implementa o algoritmo DFreeMO.
"""
function _DFreeMO_optimize!(result::OptimResult, f!::Function, Jf!::Function, h!::Function, m::Int, subproblem::Function, options::DFreeMOParams, callback::Function)
    
    # Initialize optimization
    x = copy(result.x)
    n = length(x)

    T = eltype(x)
    Fx = zeros(T, m)
    JF = zeros(T, m, n)
    h = zeros(T, m)

    # Initialize B matrices as identity for each objective
    B = [Matrix{T}(I, n, n) for _ in 1:m]
    
    # Initialize sigma
    sigma = options.sigma

    # Initialize lambda
    scale = options.epsilon / (options.sigma*sqrt(n))
    lambda = ifelse(
        isfinite(scale) && scale > 0,
        max(scale * rand(T), options.lambda_min),
        options.lambda_min
    )

    # Início da contagem de tempo para verificação de tempo máximo
    start_time = time()

    # Define the composite objective function F(x) = f(x) + h(x)
    function F!(x; m::Int=m)
        Fx = zeros(T, m)
        h = zeros(T, m)
        
        # Usar try-catch para capturar qualquer erro durante a avaliação
        try
            f!(Fx, x)
            h!(h, x)
            
            # Verificar se h! falhou
            if !result.success
                # Em vez de lançar erro, retornar um código especial
                # que será verificado em armijo
                return nothing, result.message
            end
            
            return Fx + h, nothing
        catch e
            # Capturar qualquer outro erro que possa ocorrer
            result.success = false
            result.message = "Error evaluating F!: " * string(e)
            return nothing, result.message
        end
    end
    
    if options.verbose > 0
        @info "----------------------------------------------------------------------"
        @info "Derivative-Free Method for Composite Multiojective Optimization"
        @info "----------------------------------------------------------------------"
        @info "N of variables=$(n), N of objectives=$(m), Opt tol=$(options.opt_tol), Max iter=$(options.max_iter)"
        @info "Epsilon=$(options.epsilon), Sigma=$(sigma), Alpha=$(options.alpha)"
    end

    for iter in 1:options.max_iter

        if time() - start_time >= options.max_time
            result.success = false
            result.message = "Maximum time reached $(options.max_time)"
            return result
        end

        # Compute F and JF
        f!(Fx, x)
        Jf!(JF, x; lambda=lambda)
        h!(h, x)
        
        # Verificar se h! falhou
        if !result.success
            break
        end
        
        result.Fval .= Fx + h

        # Solve the subproblem with multiple iterations
        start_subproblem_time = time()
        x_prev = copy(x)
        F_prev = copy(result.Fval)
        
        for sub_iter in 1:options.max_subproblem_iter
            theta, p = subproblem(x, JF, h, :dfreemo; sigma=sigma, B=B)
            result.subproblem_time += time() - start_subproblem_time
            
            # WARNING: This is not in the original paper
            # Se o subproblema falhou, multiplica sigma por 2 e continua
            # if !result.success || !isfinite(theta) || any(x -> !isfinite(x), p)
            #     sigma = sigma * 2
            #     # Reset success flag to allow retry
            #     result.success = true
            #     continue
            # end

            if !result.success
                break
            end

            result.abs_theta = abs(theta)

            # Check convergence
            if sigma * norm(p - x) <= options.epsilon
                x = p
                if options.verbose > 0
                    @info "Converged at iteration $iter with small tolerance"
                end
                result.success = true
                result.message = "Solution was found"
                # Update result
                norm_progress = norm(x - result.x, Inf) / max(1, norm(result.x, Inf))
                update!(result, x=x, norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)
                return result
            end

            # Check sufficient decrease
            # TODO: Check if is possible not create F_new and h_new (high memorycost)
            F_new = zeros(T, m)
            h_new = zeros(T, m)
            f!(F_new, p)
            h!(h_new, p)

            # Verificar se h! falhou
            if !result.success
                break
            end

            F_new .= F_new + h_new
            
            if all(F_new .<= F_prev .- (options.alpha * options.epsilon^2) / (2 * sigma))
                # Update x and matrices
                x_prev = copy(x)
                x = p
                JF_prev = copy(JF)
                Jf!(JF, x; lambda=lambda)
                
                # Update B matrices using BFGS
                BFGS_update!(B, n, m, x, x_prev, JF, JF_prev)
                break
            else
                sigma = sigma * 2
                # Update lambda
                scale = options.epsilon / (sigma*sqrt(n))
                lambda = ifelse(isfinite(scale) && scale > 0, max(scale * rand(T), options.lambda_min), options.lambda_min)
                Jf!(JF, x; lambda=lambda)
                # QUESTION: If the gradient is improved, why not use it to update the B matrices?
            end

            if sub_iter >= options.max_subproblem_iter
                result.success = false
                result.message = "Subproblem did not converge"
                break
            end
        end

        if options.verbose > 0
            print_table(result)
        end

        # Update result
        norm_progress = norm(x - result.x, Inf) / max(1, norm(result.x, Inf))
        update!(result, x=x, norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)

        # Stopping criteria
        #TODO: This stop criteria need to be improved
        if norm_progress <= options.opt_tol
            if options.stop_criteria == :condg
                theta, p = subproblem(x, JF, h, :condg)
                if !result.success || abs(theta) <= options.opt_tol
                    result.message = "The solution was found using |θ_CondG| <= $(options.opt_tol)"
                    break
                end
            elseif options.stop_criteria == :proxgrad
                theta, p = subproblem(x, JF, h, :proxgrad)
                if !result.success || abs(theta) <= options.opt_tol
                    result.message = "The solution was found using |θ_ProxGrad| <= $(options.opt_tol)"
                    break
                end
            elseif options.stop_criteria == :dfreemo &&  result.abs_theta <= options.opt_tol
                result.message = "The solution was found using |θ_DFreeMO| <= $(options.opt_tol)"
                break
            end
        end

        # Update lambda
        scale = options.epsilon / (sigma*sqrt(n))
        lambda = ifelse(
            isfinite(scale) && scale > 0,
            max(scale * rand(T), options.lambda_min),
            options.lambda_min
        )

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
