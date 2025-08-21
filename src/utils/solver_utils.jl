"""
    OptimResult{T<:AbstractFloat}

Container structure holding the results produced by an optimization solver.

# Fields
- `x::Vector{T}`: Final iterate returned by the solver.
- `Fval::Vector{T}`: Value of the composite objective `F(x)` at `x`.
- `x_init::Vector{T}`: Initial point provided to the solver.
- `F_init::Vector{T}`: Value of the composite objective at `x_init`.
- `iter::Int`: Number of iterations performed.
- `abs_theta::T`: Absolute value of the optimality measure θ at the last iterate.
- `norm_progress::T`: Normalised ∞-norm progress between consecutive iterates.
- `n_f_evals::Int`: Number of objective evaluations (`f!`).
- `n_Jf_evals::Int`: Number of Jacobian evaluations (`Jf!`).
- `total_time::Float64`: Total wall-clock time spent by the solver (seconds).
- `subproblem_time::Float64`: Cumulative time spent solving sub-problems (seconds).
- `x_trace::Vector{Vector{T}}`: History of iterates (populated when `store_trace=true`).
- `abs_theta_trace::Vector{T}`: History of `abs_theta` values when tracing is enabled.
- `norm_progress_trace::Vector{T}`: History of `norm_progress` values when tracing is enabled.
- `success::Bool`: `true` if the solver terminated successfully, `false` otherwise.
- `message::String`: Human-readable termination or error message.
"""
mutable struct OptimResult{T<:AbstractFloat}
    x::Vector{T}                 # current point
    Fval::Vector{T}              # current F(x)
    x_init::Vector{T}            # initial point
    F_init::Vector{T}            # F(x_init)
    iter::Int
    abs_theta::T
    norm_progress::T
    n_f_evals::Int
    n_Jf_evals::Int
    total_time::Float64
    subproblem_time::Float64
    x_trace::Vector{Vector{T}}
    abs_theta_trace::Vector{T}
    norm_progress_trace::Vector{T}
    success::Bool
    message::String
end

function OptimResult(x0::AbstractVector{T}, Fval::AbstractVector{T}) where {T<:AbstractFloat}
    return OptimResult{T}(
        copy(x0),               # x
        copy(Fval),             # Fval
        copy(x0),               # x_init
        copy(Fval),             # F_init
        0,                      # iter
        T(Inf),                 # abs_theta
        T(Inf),                 # norm_progress
        0,                      # n_f_evals
        0,                      # n_Jf_evals
        0.0,                      # total_time
        0.0,                      # subproblem_time
        Vector{Vector{T}}(),    # x_trace
        T[],                    # abs_theta_trace
        T[],                    # norm_progress_trace
        true,                   # success
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
        result.abs_theta = theta
        if store_trace
            push!(result.abs_theta_trace, theta)
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
- `lb::AbstractVector{T}`: limites inferiores das variáveis
- `ub::AbstractVector{T}`: limites superiores das variáveis
"""
struct SubProblemParams{T<:AbstractFloat}
    A::Vector{<:AbstractMatrix{T}}
    b::Vector{<:AbstractVector{T}}
    lb::AbstractVector{T}
    ub::AbstractVector{T}
    
    # Construtor interno com validações
    function SubProblemParams(A::Vector{<:AbstractMatrix{T}}, b::Vector{<:AbstractVector{T}}, lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:AbstractFloat}
        # Validações básicas
        @assert length(A) == length(b) "Número de matrizes A deve ser igual ao número de vetores b"
        @assert length(lb) == length(ub) "Limites inferiores e superiores devem ter o mesmo tamanho"
        
        # Validação adicional: verificar se as dimensões são compatíveis
        if !isempty(A) && !isempty(b)
            n_vars = size(A[1], 2)  # número de colunas da primeira matriz A
            @assert length(lb) == n_vars "Dimensão dos limites deve ser igual ao número de variáveis"
            @assert length(ub) == n_vars "Dimensão dos limites deve ser igual ao número de variáveis"
            
            # Verificar se todas as matrizes A têm o mesmo número de colunas
            for (i, A_i) in enumerate(A)
                @assert size(A_i, 2) == n_vars "Todas as matrizes A devem ter o mesmo número de colunas (variáveis)"
            end
        end
        
        return new{T}(A, b, lb, ub)
    end
end

function evalh!(h::AbstractVector{T}, x::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)
    
    # Check if any SubProbParams.b[j] has any zero value
    has_zero = false
    for j = 1:m
        if any(SubProbParams.b[j] .== 0)
            has_zero = true
            break
        end
    end
    
    # If any b[j] has a zero value, set all h to zero
    if has_zero
        fill!(h, zero(T))
        return true, "All h set to zero due to zero values in b"
    end
    
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
function process_bounds(bound, n::Int, default_value)
    if bound === nothing
        return fill(default_value, n)
    elseif isa(bound, Number)
        T = typeof(default_value)
        return fill(convert(T, bound), n)
    else
        return bound
    end
end

function print_table(result::OptimResult)
    if result.iter == 0
        separator = "-"^70
        @info separator
        headers = ["Iter", "|θ(x^k)|", "    ||x^k - x^{k-1}||/max{1, ||x^{k-1}||}"]
        n_fields = length(headers)
        format_str = "%7s  %14s  %25s"
        fmt = Printf.Format(format_str)
        header_str = Printf.format(fmt, headers...)
        
        @info header_str
        @info separator
        
        # Print progress row
        progress_str = @sprintf("%5d  %18.6e  %25.6e", 
                result.iter, result.abs_theta, 0.0)
        @info progress_str
    else
        # Print progress row
        progress_str = @sprintf("%5d  %18.6e  %25.6e", 
                result.iter, result.abs_theta, result.norm_progress)
        @info progress_str
    end
end

"""
    BFGS_update!(B, n, m, x, x_prev, JF, JF_prev)

Update B matrices using BFGS formula for each objective.

# Arguments:
- `B::Vector{<:AbstractMatrix{T}}`: BFGS matrices to be updated in-place
- `n::Int`: dimension of the input space
- `m::Int`: number of objective functions
- `x::AbstractVector{T}`: current point
- `x_prev::AbstractVector{T}`: previous point
- `JF::AbstractMatrix{T}`: current Jacobian (m x n)
- `JF_prev::AbstractMatrix{T}`: previous Jacobian (m x n)

# Returns:
- `B`: updated BFGS matrices
"""
function BFGS_update!(B::Vector{<:AbstractMatrix{T}}, n::Int, m::Int, 
                     x::AbstractVector{T}, x_prev::AbstractVector{T}, 
                     JF::AbstractMatrix{T}, JF_prev::AbstractMatrix{T}) where {T<:AbstractFloat}
    
    # Vetor s: diferença entre pontos consecutivos
    s = x - x_prev
    
    # Vetores temporários para cálculos intermediários
    Bs = Vector{T}(undef, n)
    
    # Atualiza as m matrizes BFGS (uma para cada função objetivo)
    for j in 1:m
        # Vetor y: diferença entre os gradientes consecutivos da função objetivo j
        y = JF[j,:] - JF_prev[j,:]
        
        # Produto interno s'y (critério de curvatura)
        s_dot_y = dot(s, y)
        
        # Somente atualiza a matriz BFGS se s'y > 0 (condição de curvatura positiva)
        if s_dot_y > zero(T)
            # Calcula B*s reusando o vetor Bs
            mul!(Bs, B[j], s)
            sBs = dot(s, Bs)
            
            # Atualiza B[j] in-place usando a fórmula BFGS
            for i in 1:n
                for k in 1:n
                    B[j][i,k] -= (Bs[i] * Bs[k]) / sBs - (y[i] * y[k]) / s_dot_y
                end
            end
        end
        # Se s'y <= 0, não faz nada, mantendo B[j] inalterado
    end
    
    return B  # retorna as matrizes atualizadas para encadeamento de chamadas
end

"""
    jacobian_finite_diff(f!, x, m; method=:central, λ=sqrt(eps(T)))

Calculate the full Jacobian using finite differences.

# Arguments:
- `f!::Function`: Function to evaluate the objective function f(x): R^n -> R^m
- `x::AbstractVector{T}`: Point where the Jacobian will be evaluated
- `m::Int`: Number of objective functions
- `method::Symbol`: Finite difference method (:forward, :backward, :central)
- `λ::T`: Step size for finite differences

# Returns:
- `J::Matrix{T}`: Full Jacobian matrix
"""
function jacobian_finite_diff(f!::Function, x::AbstractVector{T}, m::Int; method::Symbol=:central, λ::T=sqrt(eps(T))) where {T<:AbstractFloat}
    # Number of variables
    n = length(x)

    # Output Jacobian (m × n)
    J = zeros(T, m, n)

    # Base evaluation f(x) – reused in forward/backward formulas
    f_base = zeros(T, m)
    f!(f_base, x)

    # Work buffers reused across iterations to avoid allocations
    f_plus  = similar(f_base)
    f_minus = similar(f_base)
    x_plus  = copy(x)   # updated in place
    x_minus = copy(x)

    if method == :forward
        for i in 1:n
            x_plus[i] += λ
            f!(f_plus, x_plus)
            @inbounds for j in 1:m
                J[j, i] = (f_plus[j] - f_base[j]) / λ
            end
            x_plus[i] = x[i]          # restore coordinate
        end

    elseif method == :backward
        for i in 1:n
            x_minus[i] -= λ
            f!(f_minus, x_minus)
            @inbounds for j in 1:m
                J[j, i] = (f_base[j] - f_minus[j]) / λ
            end
            x_minus[i] = x[i]
        end

    elseif method == :central
        half_inv = T(1) / (2*λ)
        for i in 1:n
            x_plus[i]  += λ
            x_minus[i] -= λ
            f!(f_plus,  x_plus)
            f!(f_minus, x_minus)
            @inbounds for j in 1:m
                J[j, i] = (f_plus[j] - f_minus[j]) * half_inv
            end
            # restore coordinates for the next iteration
            x_plus[i]  = x[i]
            x_minus[i] = x[i]
        end
    else
        error("Invalid finite-difference method: $method. Choose :forward, :backward or :central.")
    end

    return J
end

#TODO: Use this function to check if the subproblem is feasible# Optional: Presolve function to check problem feasibility
function check_subproblem_feasibility(x::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}
    n = length(x)
    
    # Check if x is within bounds
    if any(x .< SubProbParams.lb) || any(x .> SubProbParams.ub)
        return false, "Initial point violates bounds"
    end
    
    # Check if constraint system has a solution
    for A in SubProbParams.A
        if size(A, 2) != n
            return false, "Dimension mismatch in constraint matrix"
        end
    end
    
    return true, "Problem appears feasible"
end

# Optional: Helper function to create a solver cache
function create_pdfpm_cache(::Type{T}) where {T<:AbstractFloat}
    model = Model(Ipopt.Optimizer)
    return model
end
