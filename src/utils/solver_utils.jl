"""
    OptimResult

Estrutura que armazena os resultados de um solver de otimização.

# Campos
- `x::Vector{Float64}`: ponto final encontrado
- `n_f_evals::Int`: número de avaliações de função
- `n_Jf_evals::Int`: número de avaliações de jacobiana
- `success::Bool`: indicador de sucesso
- `message::String`: mensagem descritiva do resultado
"""
mutable struct OptimResult{T<:AbstractFloat}
    x::Vector{T}
    iter::Int
    theta::T
    norm_progress::T
    n_f_evals::Int
    n_Jf_evals::Int
    total_time::Float64
    subproblem_time::Float64
    x_trace::Vector{Vector{T}}
    theta_trace::Vector{T}
    norm_progress_trace::Vector{T}
    success::Bool
    message::String
end

function OptimResult(x0::AbstractVector{T}) where {T<:AbstractFloat}
    return OptimResult{T}(
        copy(x0),
        0,                        # iter
        T(Inf),                   # theta
        T(Inf),                   # norm_progress
        0,                        # n_f_evals
        0,                        # n_Jf_evals
        0,                        # total_time
        0,                        # subproblem_time
        Vector{Vector{T}}(),      # x_trace
        T[],                      # theta_trace
        T[],                      # norm_progress_trace
        true,                     # success
        "Optimization not started")  # message
end

function update!(result::OptimResult{T}; x=nothing, theta=nothing, norm_progress=nothing, iter=nothing, store_trace=false) where {T<:AbstractFloat}
    if x !== nothing
        result.x = x
        if store_trace
            push!(result.x_trace, x)
        end
    end

    if theta !== nothing
        result.theta = theta
        if store_trace
            push!(result.theta_trace, theta)
        end
    end

    if norm_progress !== nothing
        result.norm_progress = norm_progress
        if store_trace
            push!(result.norm_progress_trace, norm_progress)
        end
    end

    if iter !== nothing
        result.iter = iter
    end
end

"""
    SubProblemParams

Estrutura que armazena os parâmetros de um subproblema.

# Campos
- `A::Vector{<:AbstractMatrix{T}}`: matrizes A do subproblema
- `b::Vector{<:AbstractVector{T}}`: vetores b do subproblema
"""

struct SubProblemParams{T<:AbstractFloat}
    A::Vector{<:AbstractMatrix{T}}
    b::Vector{<:AbstractVector{T}}
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

# Em solver_utils.jl (se ainda não existir)
function SubProblemParams(A::Vector{<:AbstractMatrix{T}}, b::Vector{<:AbstractVector{T}}, lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:AbstractFloat}
    # Validações básicas
    @assert length(A) == length(b) "Número de matrizes A deve ser igual ao número de vetores b"
    return SubProblemParams{T}(A, b, lb, ub)
end

function evalh!(h::AbstractVector{T}, x::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)
    
    for j = 1:m
        # Create a new model for each objective
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        
        # Define variables
        @variable(model, z[1:n])

        # Define objective function (maximize x'z)
        @objective(model, Max, dot(x, z))

        # Add constraints
        @constraint(model, SubProbParams.A[j] * z .<= SubProbParams.b[j])

        # Solve the problem
        optimize!(model)

        # Check if solution was found (accept multiple success statuses)
        if termination_status(model) == MOI.OPTIMAL || 
           termination_status(model) == MOI.LOCALLY_SOLVED || 
           termination_status(model) == MOI.ALMOST_OPTIMAL || 
           termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
            h[j] = objective_value(model)
        else
            @warn "ERROR: Linear program could not be solved! Status: $(termination_status(model))"
            return false, "Failed evaluating h_$j(x^k)."
        end
    end

    return true, "All subproblems solved successfully"
end

# Função auxiliar para processar limites
function process_bounds(bound, n::Integer, default_value)
    if bound === nothing
        return fill(default_value, n)
    elseif isa(bound, Number)
        return fill(convert(Float64, bound), n) # TODO: change to T
    else
        return bound
    end
end

function print_table(result::OptimResult)
    if result.iter == 0
        separator = "-"^70
        @info separator
        headers = ["Iteration", "theta", "||x_k - x_{k-1}||_∞/max(1, ||x_{k-1}||_∞)"]
        n_fields = length(headers)
        format_str = join(["%$(10+i*3)s" for i in 0:(n_fields-1)], "  ")
        fmt = Printf.Format(format_str)
        header_str = Printf.format(fmt, headers...)
        
        @info header_str
        @info separator
        
        # Print progress row
        progress_str = @sprintf("%6d  %18.6e  %16.6e", 
                result.iter, result.theta, 0.0)
        @info progress_str
    else
        # Print progress row
        progress_str = @sprintf("%6d  %18.6e  %16.6e", 
                result.iter, result.theta, result.norm_progress)
        @info progress_str
    end
end

# """
#     SolverResult

# Estrutura que armazena os resultados de um solver de otimização multiobjetivo.

# # Campos
# - `pareto_front::Vector{Vector{Float64}}`: conjunto de pontos não-dominados no espaço de objetivos
# - `pareto_set::Vector{Vector{Float64}}`: conjunto de pontos não-dominados no espaço de variáveis
# - `iterations::Int`: número de iterações realizadas
# - `success::Bool`: indicador de sucesso
# - `message::String`: mensagem descritiva do resultado
# """
# struct SolverResult
#     pareto_front::Vector{Vector{Float64}}
#     pareto_set::Vector{Vector{Float64}}
#     iterations::Int
#     success::Bool
#     message::String
# end

# """
#     get_pareto_front(result::SolverResult)

# Retorna o conjunto de pontos não-dominados no espaço de objetivos.
# """
# get_pareto_front(result::SolverResult) = result.pareto_front

# """
#     get_pareto_set(result::SolverResult)

# Retorna o conjunto de pontos não-dominados no espaço de variáveis.
# """
# get_pareto_set(result::SolverResult) = result.pareto_set

# """
#     is_dominated(f1, f2)

# Verifica se o ponto f1 é dominado por f2 em um problema de minimização.
# Retorna true se f2 domina f1, false caso contrário.
# """
# function is_dominated(f1::Vector{Float64}, f2::Vector{Float64})
#     all_leq = true
#     any_less = false
    
#     for i in 1:length(f1)
#         if f1[i] < f2[i]
#             all_leq = false
#             break
#         elseif f1[i] > f2[i]
#             any_less = true
#         end
#     end
    
#     return any_less && all_leq
# end

# """
#     update_pareto_set!(pareto_front, pareto_set, f_new, x_new)

# Atualiza o conjunto de Pareto com um novo ponto.
# Retorna true se o ponto foi adicionado, false caso contrário.
# """
# function update_pareto_set!(pareto_front::Vector{Vector{Float64}}, 
#                            pareto_set::Vector{Vector{Float64}}, 
#                            f_new::Vector{Float64}, 
#                            x_new::Vector{Float64})
#     # Verifica se o novo ponto é dominado por algum ponto existente
#     for f in pareto_front
#         if is_dominated(f_new, f)
#             return false
#         end
#     end
    
#     # Remove pontos dominados pelo novo ponto
#     i = 1
#     while i <= length(pareto_front)
#         if is_dominated(pareto_front[i], f_new)
#             deleteat!(pareto_front, i)
#             deleteat!(pareto_set, i)
#         else
#             i += 1
#         end
#     end
    
#     # Adiciona o novo ponto
#     push!(pareto_front, copy(f_new))
#     push!(pareto_set, copy(x_new))
    
#     return true
# end
