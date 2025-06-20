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
# - `stop_criteria::Symbol = :proxgrad`:  Additional stop criteria (:condg, :proxgrad, :dfreemo)
# - `store_trace::Bool = false`: Whether to store the trace of x during optimization
"""
function CondG_options(; verbose=1, max_iter=1000, opt_tol=1e-4, ftol=1e-4, max_time=3600.0, print_interval=10, stop_criteria=:proxgrad, store_trace=false)
    if stop_criteria ∉ [:condg, :proxgrad, :dfreemo]
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
# - `callback::Function`: Callback function to be called at each iteration
"""


function CondG(evalf::Function, evalJf::Function, A::Vector{<:AbstractMatrix{T}}, delta::T, 
    x0::AbstractVector{T}, options::CondGParams=CondG_options(); 
    lb::Union{Nothing,AbstractVector{T},Number}=nothing, 
    ub::Union{Nothing,AbstractVector{T},Number}=nothing,
    callback=noop_callback) where {T<:AbstractFloat}

    start_time = time()

    n = length(x0)
    m = length(A)
    A2n = [A[i]; -A[i] for i in 1:m]
    b2n = [delta * ones(2*n) for i in 1:m]

    # Validações
    @assert m > 0 "Pelo menos uma matriz A deve ser fornecida"

    for i in 1:m
        @assert size(A[i], 1) == n && size(A[i], 2) == n "A matriz A[$i] deve ser quadrada de tamanho $n×$n"
    end

    # Processamento dos limites
    lb_vec = process_bounds(lb, n, -Inf)
    ub_vec = process_bounds(ub, n, Inf)

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
    result = OptimResult(x0)

    function f!(f, x)
        result.n_f_evals += 1
        f = evalf(x)
        return f
    end

    function Jf!(Jf, x)
        result.n_Jf_evals += 1
        Jf = evalJf(x)
        return Jf
    end

    """	
    Create the function
                    h(x): R^n -> R^m,
                    h(x) = (h_1(x), ..., h_m(x)),
    where h_j(x) := max{<x,z> | z ∈ Z_j} and Z_j := {z ∈ R^n | -δ·e ≤ A_j z ≤ δ·e}  for j = 1, ..., m
    """

    function h!(h, x)
        #WARNING: The evaluation of function h! can be failed in line search subroutine 
        success, message = evalh!(h, x, SubProbParams)
        if !success
            result.success = false
            result.message = "evalh! failed: " * message
        end
        return h
    end

    function subproblem(x, JF, h, method::Symbol)
        if method == :condg
            theta, p, success, message = subproblem_condg(x, JF, h, SubProbParams)
            if !success
                result.success = false
                result.message = "The subproblem of CondG failed: " * message #QUESTION: Porque * message?
            end
            return theta, p
        elseif method == :proxgrad
            theta, p, success, message = subproblem_proxgrad(x, JF, h, SubProbParams)
            if !success
                result.success = false
                result.message = "The subproblem of ProxGrad failed: " * message
            end
            return theta, p
        elseif method == :dfreemo
            theta, p, success, message = subproblem_dfreemo(x, JF, h, SubProbParams)
            if !success
                result.success = false
                result.message = "The subproblem of DFreeMO failed: " * message
            end
            return theta, p
        else
            error("The subproblem method is not supported: $method")
        end
    end




    _CondG_optimize!(result, f!, Jf!, h!, subproblem, options, callback)
    result.total_time = time() - start_time
    return result
end

"""
    _CondG_optimize!(f!, Jf!, h!, x0, options, callback, result)

Função interna que implementa o algoritmo CondG.
"""
function _CondG_optimize!(result::OptimResult, f!::Function, Jf!::Function, h!::Function, subproblem::Function, options::CondGParams, callback::Function) where {T<:AbstractFloat}
    
    # Define the composite objective function F(x) = f(x) + h(x)
    function F!(F, x)
        F = f!(F, x) + h!(F, x)
        return F
    end

    # Initialize optimization
    x = copy(result.x)
    n = length(x)
    m = length(subproblem.A)
    JF = zeros(T, m, n)
    h = zeros(T, m)
    
    if options.verbose > 0
        @info "----------------------------------------------------------------------"
        @info "Conditional Gradient Method for Composite Multiojective Optimization"
        @info "----------------------------------------------------------------------"
        @info "Parameters: Number of variables=$(n), Number of objectives=$(m), Optimality tolerance=$(options.opt_tol), Max iterations=$(options.max_iter)"
    end

    if options.verbose > 0
        print_table(result)
    end

    for iter in 1:options.max_iter
        # Compute the Jacobian of f
        Jf!(JF, x)
        h!(h, x)
        if !result.success
            break
        end

        # Solve the subproblem
        start_subproblem_time = time()
        theta, p = subproblem(x, JF, h, :condg)
        result.subproblem_time += time() - start_subproblem_time
        if !result.success
            break
        end

        # Define the search direction
        d = p - x

        # Compute the step size
        line_search_result = armijo(F!, x, d, Fx, Jf, theta)

        if !line_search_result.success
            result.message = line_search_result.message
            result.success = false
            break
        end

        x = x + line_search_result.stp * d

        norm_progress = norm(x - result.x, Inf) / max(1, norm(result.x, Inf))

        # Update
        update!(result, x=x, theta=theta, norm_progress=norm_progress, iter=iter, store_trace=options.store_trace)

        if options.verbose > 0
            print_table(result)
        end

        # Stopping criteria
        # Test optimality
        if norm_progress <= options.opt_tol
            if options.stop_criteria == :condg && abs(theta) <= options.opt_tol
                break
            elseif options.stop_criteria == :proxgrad
                theta, p = subproblem(x, JF, h, :proxgrad)
                if !result.success || abs(theta) <= options.opt_tol
                    break
                end
            elseif options.stop_criteria == :dfreemo
                theta, p = subproblem(x, JF, h, :dfreemo)
                if !result.success || abs(theta) <= options.opt_tol
                    break
                end
            end
        end
    end
    
    if options.verbose > 0 && result.iter >= options.max_iter
        result.message = "Maximum number of iterations reached $(options.max_iter)"
        @info result.message
        result.success = false
    end
    
    return result
end

"""
    noop_callback

Callback padrão que não faz nada.
"""
function noop_callback(args...)
    # Não faz nada
    return nothing
end



























# # solvers/condg.jl
# function solve_condg(
#     model::MOCompositeModel{T, S},
#     options::Dict{Symbol, Any} = Dict{Symbol, Any}()
# ) where {T, S}
    
#     # Extrair parâmetros do modelo
#     n = model.meta.nvar
#     m = model.nobj
#     x = copy(model.meta.x0)
#     lvar = model.meta.lvar
#     uvar = model.meta.uvar
    
#     # Configurar opções
#     tol = get(options, :tol, 1e-4)
#     ftol = get(options, :ftol, 1e-4)
#     max_iter = get(options, :max_iter, 100)
    
#     # Inicializar contadores
#     iter = 0
#     nfev = 0
#     ngev = 0
    
#     # Avaliar funções iniciais
#     f_vals = zeros(T, m)
#     g_vals = zeros(T, m)
#     for i = 1:m
#         f_vals[i] = eval_f(model, x, i)
#         g_vals[i] = eval_g(model, x, i)
#     end
#     F_vals = f_vals + g_vals
#     nfev += 1
    
#     # Loop principal
#     while iter < max_iter
#         # Calcular gradientes
#         Jf = zeros(T, m, n)
#         for i = 1:m
#             Jf[i,:] = eval_grad_f(model, x, i)
#         end
#         ngev += 1
        
#         # Resolver subproblema
#         tau, p, flag = subproblem_condg(n, m, lvar, uvar, x, Jf, g_vals)
        
#         # Verificar critério de parada
#         if abs(tau) < tol
#             return Dict(
#                 :solution => x,
#                 :f_values => f_vals,
#                 :g_values => g_vals,
#                 :status => :success,
#                 :iterations => iter,
#                 :f_evaluations => nfev,
#                 :g_evaluations => ngev
#             )
#         end
        
#         # Calcular direção
#         d = p - x
        
#         # Busca linear
#         alpha, new_f_vals, new_g_vals = armijo_line_search(model, x, d, F_vals, Jf, tau, ftol)
#         nfev += 1
        
#         # Atualizar ponto
#         x = x + alpha * d
#         f_vals = new_f_vals
#         g_vals = new_g_vals
#         F_vals = f_vals + g_vals
        
#         iter += 1
#     end
    
#     # Máximo de iterações atingido
#     return Dict(
#         :solution => x,
#         :f_values => f_vals,
#         :g_values => g_vals,
#         :status => :max_iter,
#         :iterations => iter,
#         :f_evaluations => nfev,
#         :g_evaluations => ngev
#     )
# end