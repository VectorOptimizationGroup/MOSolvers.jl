# function subproblem_proxgrad(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, SubProbParams::SubProblemParams; sigma::T=1.0) where {T<:AbstractFloat}

#     n = length(x)
#     m = length(SubProbParams.A)

#     # Create the model
#     model = Model(Ipopt.Optimizer)
    
#     # Configurações específicas para o Ipopt para melhorar a convergência
#     set_optimizer_attribute(model, "max_iter", 3000)
#     set_optimizer_attribute(model, "tol", 1e-6)
#     set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
#     set_optimizer_attribute(model, "acceptable_tol", 1e-4)
#     set_optimizer_attribute(model, "print_level", 0)
    
#     # Define variables
#     @variable(model, tau)
#     @variable(model, SubProbParams.lb[i] <= u[i=1:n] <= SubProbParams.ub[i])  # Variável u com limites

#     if minimum(h) != 0.0
#         # Create variables w_j for each objective function
#         w = Dict()
#         for j in 1:m
#             w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)
#         end
#     end
    
#     # Define objective function: tau + (sigma/2)||u - x||^2
#     @objective(model, Min, tau + 0.5 * sigma * sum((u[i] - x[i])^2 for i in 1:n))

#     # Restrições
#     if minimum(h) != 0.0
#         for j in 1:m
#             # Restrição: <b_j, w_j> - h_j(x) + <∇f_j(x), u - x> ≤ tau
#             @constraint(model, 
#                 dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], u .- x) <= tau)
            
#             # Restrição: A_j^T w_j = u
#             for i in 1:n
#                 @constraint(model, 
#                     sum(SubProbParams.A[j][r, i] * w[j][r] for r in 1:size(SubProbParams.A[j], 1)) == u[i])
#             end
#         end
#     else
#         for j in 1:m
#             # Restrição: <∇f_j(x), u - x> ≤ tau
#             @constraint(model, 
#                 dot(Jf[j,:], u .- x) <= tau)
#         end
#     end

#     # Solve the problem
#     optimize!(model)

#     # Check if solution was found (accept multiple success statuses)
#     if termination_status(model) == MOI.OPTIMAL || 
#         termination_status(model) == MOI.LOCALLY_SOLVED || 
#         termination_status(model) == MOI.ALMOST_OPTIMAL || 
#         termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
#         success = true
#         message = "Nonlinear program was solved!"
#         tau_val = value(tau)
#         u_val = [value(u[i]) for i in 1:n]
#         theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
    
#     else
#         @warn "ERROR: Nonlinear program could not be solved! Status: $(termination_status(model))"
#         tau_val = convert(T, NaN)
#         u_val = fill(convert(T, NaN), n)
#         theta = convert(T, NaN)
#         success = false
#         message = "Nonlinear program could not be solved! Status: $(termination_status(model))"
#     end

#     return theta, u_val, success, message
# end

# function subproblem_proxgrad(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, 
#     SubProbParams::SubProblemParams; 
#     sigma::T=1.0,
#     tol::T=sqrt(eps(T)),
#     max_iter::Int=3000,
#     verbose::Int=0,
#     warm_start::Union{Nothing, AbstractVector{T}}=nothing,
#     solver_cache::Union{Nothing, Model}=nothing) where {T<:AbstractFloat}

#     n = length(x)
#     m = length(SubProbParams.A)

#     # Enhanced input validation with detailed error checking
#     if !isfinite(sigma) || sigma < sqrt(eps(T)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid input parameters"
#     end

#     # Validate constraint matrices with numerical stability checks
#     for (j, Aj) in enumerate(SubProbParams.A)
#         # Check for numerical issues in constraint matrices
#         if any(!isfinite, Aj)
#             return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid A matrix at index $j"
#         end
#         # Check for rank deficiency which could cause solver issues
#         if rank(Aj) < min(size(Aj)...)
#             @warn "Constraint matrix A[$j] may be rank deficient"
#         end
#     end
    
#     # Create or reuse model for better performance in repeated calls
#     if solver_cache === nothing
#         model = Model(Ipopt.Optimizer)
#     else
#         model = solver_cache
#         # Clear previous solution to avoid contamination
#         empty!(model)
#     end

#     # Adaptive Ipopt settings based on precision type and tolerance requirements
#     set_optimizer_attribute(model, "max_iter", max_iter)
#     set_optimizer_attribute(model, "tol", max(tol, 1e-14))  # Ipopt has minimum tolerance limits
#     set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
#     set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))  # More relaxed backup tolerance
#     set_optimizer_attribute(model, "print_level", verbose)
    
#     # Numerical stability settings that help with ill-conditioned problems
#     set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
#     set_optimizer_attribute(model, "bound_relax_factor", max(tol/100, 1e-12))
#     set_optimizer_attribute(model, "honor_original_bounds", "yes")
    
#     # Additional robustness settings for difficult problems
#     set_optimizer_attribute(model, "mu_strategy", "adaptive")
#     set_optimizer_attribute(model, "jacobian_regularization_value", max(eps(T), 1e-8))
    
#     # Define variables with adaptive bounds to prevent numerical overflow
#     bound_scale = T(1e6) * max(one(T), maximum(abs, x))  # Scale bounds with problem magnitude
#     lb_safe = max.(SubProbParams.lb, -bound_scale)
#     ub_safe = min.(SubProbParams.ub, bound_scale)
    
#     # Check for feasibility issues before proceeding
#     if any(lb_safe .> ub_safe)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Infeasible bounds"
#     end
    
#     # Define variables
#     @variable(model, tau)
#     @variable(model, lb_safe[i] <= u[i=1:n] <= ub_safe[i])  # Use safe bounds
    
#     # Warm start capability for better convergence when good initial guess available
#     if warm_start !== nothing && length(warm_start) == n
#         for i in 1:n
#             if lb_safe[i] <= warm_start[i] <= ub_safe[i]
#                 set_start_value(u[i], warm_start[i])
#             end
#         end
#     end
    
#     # Use auxiliary variables for numerical stability in distance calculations
#     @variable(model, d[i=1:n])
#     @constraint(model, [i=1:n], d[i] == u[i] - x[i])

#     # Define objective function with auxiliary variables for better conditioning
#     @objective(model, Min, tau + 0.5 * sigma * sum(d[i]^2 for i in 1:n))

#     # Create variables w_j with adaptive bounds for numerical stability
#     w = Dict()

#     # TODO: test if this is the best bound for w
#     # w_bound = T(1e4) * max(one(T), maximum(abs, h))
#     # for j in 1:m
#     #     num_constraints = size(SubProbParams.A[j], 1)
#     #     w[j] = @variable(model, [1:num_constraints], lower_bound=0, upper_bound=w_bound)
#     # end
    
#     # Use unbounded variables and let Ipopt handle scaling for robustness
#     for j in 1:m
#         w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)

#         if minimum(h) != 0.0
#             @constraint(model, dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], d) <= tau)
#         else
#             @constraint(model, dot(Jf[j,:], d) <= tau)
#         end

#         # Optimized constraint generation with sparsity awareness for better performance
#         A_j = SubProbParams.A[j]
#         for i in axes(A_j, 2)
#             # Build expression efficiently based on matrix sparsity
#             expr = AffExpr()
            
#             if issparse(A_j)
#                 # For sparse matrices, iterate only over non-zero elements
#                 for k in nzrange(A_j, i)
#                     row_idx = rowvals(A_j)[k]
#                     value = nonzeros(A_j)[k]
#                     expr += value * w[j][row_idx]
#                 end
#             else
#                 # For dense matrices, skip explicit zeros for efficiency
#                 for r in axes(A_j, 1)
#                     if !iszero(A_j[r, i])
#                         expr += A_j[r, i] * w[j][r]
#                     end
#                 end
#             end
            
#             # Add the constraint with proper handling of zero expressions
#             if iszero(expr)
#                 @constraint(model, u[i] == 0.0)
#             else
#                 @constraint(model, expr == u[i])
#             end
#         end
#     end
    
#     # Solve the problem
#     optimize!(model)

#     # Enhanced solution status checking with more comprehensive success criteria
#     status = termination_status(model)

#     if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
#         # Extract solution and compute objective value consistently
#         tau_val = value(tau)
#         u_val = [value(u[i]) for i in 1:n]
#         theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
        
#         return theta, u_val, true, string(status)
#     else
#         # Enhanced error handling with informative messages
#         return convert(T, NaN), fill(convert(T, NaN), n), false, string(status)
#     end
# end

"""
    subproblem_proxgrad(x, Jf, h, SubProbParams;
                        sigma=1.0, tol=sqrt(eps(T)),
                        max_iter=3000, max_time=5.0,
                        verbose=0, warm_start=nothing,
                        solver_cache=nothing)
      -> (θ, u, success, msg)

Resolve o subproblema proximal, no ponto `x`:

    min τ + (σ/2)‖u - x‖²
    s.a.  ⟨∇f_j(x), u - x⟩ + ( ⟨b_j, w_j⟩ - h_j ) ≤ τ,  se h_j ≉ 0
          ⟨∇f_j(x), u - x⟩                        ≤ τ,  se h_j ≈ 0
          A_j' w_j = u                  (∀ j)
          lb ≤ u ≤ ub,  w_j ≥ 0

Parâmetros:
- `sigma` (σ) regulariza o objetivo (proximidade a `x`).
- `tol`, `max_iter`, `max_time`, `verbose`: atributos do Ipopt.
- `warm_start`: chute inicial para `u` (respeitado se ∈ [lb,ub]).
- `solver_cache`: reuso de `Model` para múltiplas chamadas.

Retorna:
- `θ = τ* + (σ/2)‖u* - x‖²`,
- `u` (solução),
- `success::Bool`,
- `msg::String` (status do solver).
"""
function subproblem_proxgrad(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T},
                             SubProbParams::SubProblemParams;
                             sigma::T = one(T),
                             tol::T = sqrt(eps(T)),
                             max_iter::Int = 3000,
                             max_time::Real = 5.0,
                             verbose::Int = 0,
                             warm_start::Union{Nothing, AbstractVector{T}} = nothing,
                             solver_cache::Union{Nothing, Model} = nothing) where {T<:AbstractFloat}

    # --- checks leves ---
    if !(isfinite(sigma) && sigma ≥ T(0)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), length(x)), false, "Invalid input"
    end

    # atalhos e eixos
    Iu = eachindex(x)
    m  = length(SubProbParams.A)
    A  = SubProbParams.A
    b  = SubProbParams.b
    lb = SubProbParams.lb
    ub = SubProbParams.ub

    # --- (Re)cria modelo ---
    model = solver_cache === nothing ? Model(Ipopt.Optimizer) : (empty!(solver_cache); solver_cache)

    # Verbosidade/tolerâncias/recursos (Ipopt)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "acceptable_tol", max(10tol, 1e-12))
    set_optimizer_attribute(model, "print_level", verbose)
    set_optimizer_attribute(model, "max_cpu_time", float(max_time))  # tempo em segundos

    # Opções estáveis e simples
    set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "honor_original_bounds", "yes")

    # --- variáveis ---
    @variable(model, tau)
    @variable(model, lb[i] <= u[i in Iu] <= ub[i])

    # direção d = u - x (apenas para expressar o termo proximal)
    @variable(model, d[i in Iu])
    @constraint(model, [i in Iu], d[i] == u[i] - x[i])

    # warm-start suave
    if warm_start !== nothing && length(warm_start) == length(Iu)
        for i in Iu
            v = warm_start[i]
            if lb[i] <= v <= ub[i]
                set_start_value(u[i], v)
            end
        end
    end

    # objetivo: tau + (σ/2)‖d‖²
    @objective(model, Min, tau + 0.5*sigma*sum(d[i]^2 for i in Iu))

    # --- restrições por objetivo ---
    for j in 1:m
        Aj = A[j]
        @assert axes(Aj, 2) == Iu "Dimensão de A[$j] incompatível com u"
        rset = axes(Aj, 1)

        # A_j' * w_j = u
        wj = @variable(model, [r in rset], lower_bound = 0.0)
        for i in axes(Aj, 2)
            @constraint(model, sum(Aj[r, i]*wj[r] for r in rset) == u[i])
        end

        # ⟨∇f_j(x), d⟩ [ + (⟨b_j, w_j⟩ - h_j) ] ≤ τ
        grad_j = view(Jf, j, axes(Jf,2))
        if !isapprox(h[j], zero(T); atol = max(tol, T(1e-14)))
            @constraint(model, dot(grad_j, [d[i] for i in Iu]) + (dot(b[j], wj) - h[j]) <= tau)
        else
            @constraint(model, dot(grad_j, [d[i] for i in Iu]) <= tau)
        end
    end

    # --- solve ---
    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        τ  = convert(T, value(tau))
        ū  = [convert(T, value(u[i])) for i in Iu]
        θ  = τ + 0.5*sigma*sum((ū[i] - x[i])^2 for i in Iu)
        return θ, ū, true, string(status)
    elseif status == MOI.INFEASIBLE
        return convert(T, Inf), [T(NaN) for _ in Iu], false, "Infeasible"
    else
        return convert(T, NaN), [T(NaN) for _ in Iu], false, string(status)
    end
end