# function subproblem_condg(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}

#     n = length(x)
#     m = length(SubProbParams.A)

#     # Create the model
#     model = Model(HiGHS.Optimizer)
#     set_silent(model)
    
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

#     # Define objective function
#     @objective(model, Min, tau)

#     if minimum(h) != 0.0
#         # Restrições
#         for j in 1:m
#             # Restrição: <b_j, w_j> - h_j(x) + <∇f_j(x), u - x> ≤ tau
#             @constraint(model, 
#                 dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], u .- x) <= tau)
            
#             # Restrição: A_j^T w_j = u
#             # TODO:Descobrir quanto é size(SubProbParams.A[j], 1)
#             for i in 1:n
#                 @constraint(model, 
#                     sum(SubProbParams.A[j][r, i] * w[j][r] for r in 1:size(SubProbParams.A[j], 1)) == u[i])
#             end
#         end
#     else
#         # Restrições
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
#         message = "Linear program was solved!"
#         tau_val = value(tau)
#         u_val = [value(u[i]) for i in 1:n]
    
#      else
#         @warn "ERROR: Linear program could not be solved! Status: $(termination_status(model))"
#         tau_val = convert(T, NaN)
#         u_val = fill(convert(T, NaN), n)
#         success = false
#         message = "Linear program could not be solved! Status: $(termination_status(model))"
#      end

#     return tau_val, u_val, success, message
# end

# function subproblem_condg(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, 
#     SubProbParams::SubProblemParams;
#     tol::T=sqrt(eps(T)),
#     max_iter::Int=10000,  # HiGHS typically needs more iterations than Ipopt
#     verbose::Int=0,
#     warm_start::Union{Nothing, AbstractVector{T}}=nothing,
#     solver_cache::Union{Nothing, Model}=nothing,
#     presolve::String="on") where {T<:AbstractFloat}

#     n = length(x)
#     m = length(SubProbParams.A)

#     # Enhanced input validation
#     if any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid input parameters"
#     end
    
#     # Validate constraint matrices with numerical stability checks
#     for (j, Aj) in enumerate(SubProbParams.A)
#         if any(!isfinite, Aj)
#             return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid A matrix at index $j"
#         end
#         # Check for rank deficiency which could indicate redundant constraints
#         if rank(Aj) < min(size(Aj)...)
#             @warn "Constraint matrix A[$j] may be rank deficient - this could indicate redundant constraints"
#         end
#     end

#     # Create or reuse model for performance optimization in repeated calls
#     if solver_cache === nothing
#         model = Model(HiGHS.Optimizer)
#     else
#         model = solver_cache
#         # Clear previous solution
#         empty!(model)
#     end
    
#     # Configure HiGHS with adaptive settings based on problem characteristics
#     if verbose == 0
#         set_silent(model)
#     else
#         # Allow HiGHS to provide output for debugging purposes
#         unset_silent(model)
#         set_optimizer_attribute(model, "output_flag", true)
#         set_optimizer_attribute(model, "log_to_console", true)
#     end
    
#     set_optimizer_attribute(model, "simplex_iteration_limit", max_iter)  # For simplex solver
#     set_optimizer_attribute(model, "ipm_iteration_limit", max_iter)      # For interior point method
    
#     # Numerical precision settings that adapt to the floating-point type being used
#     # This ensures we maintain appropriate accuracy for Float32, Float64, etc.
#     primal_feasibility_tol = max(tol, T(1e-12))
#     dual_feasibility_tol = max(tol, T(1e-12))
    
#     # HiGHS only accepts Float64, so we convert to Float64
#     set_optimizer_attribute(model, "primal_feasibility_tolerance", Float64(primal_feasibility_tol))
#     set_optimizer_attribute(model, "dual_feasibility_tolerance", Float64(dual_feasibility_tol))
    
#     # Presolve can significantly improve performance but might cause issues with some problems
#     # We make it configurable so users can disable it if needed
#     set_optimizer_attribute(model, "presolve", presolve)
    
#     # Advanced HiGHS settings for numerical robustness
#     set_optimizer_attribute(model, "parallel", "on")  # Use multiple cores when available
#     set_optimizer_attribute(model, "simplex_scale_strategy", 1)  # Enable scaling for better numerics
    
#     # Define variables with enhanced bound safety
#     # We create adaptive bounds that prevent numerical overflow while respecting user constraints
#     bound_scale = T(1e8) * max(one(T), maximum(abs, x))  # Scale bounds with problem magnitude
#     lb_safe = max.(SubProbParams.lb, -bound_scale)
#     ub_safe = min.(SubProbParams.ub, bound_scale)
    
#     # Early feasibility check - better to fail fast with clear message
#     if any(lb_safe .> ub_safe)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Infeasible bounds detected"
#     end
    
#     # Define decision variables with safe bounds
#     @variable(model, tau)
#     @variable(model, lb_safe[i] <= u[i=1:n] <= ub_safe[i])  # Use numerically safe bounds

#     # Warm start implementation
#     if warm_start !== nothing && length(warm_start) == n
#         # Validate warm start point lies within feasible region before using it
#         for i in 1:n
#             if lb_safe[i] <= warm_start[i] <= ub_safe[i]
#                 set_start_value(u[i], warm_start[i])
#             else
#                 # If warm start violates bounds, project it onto feasible region
#                 projected_value = max(lb_safe[i], min(warm_start[i], ub_safe[i]))
#                 set_start_value(u[i], projected_value)
#             end
#         end
#     end

#     # Create auxiliary variables for numerical stability and clearer problem structure
#     @variable(model, d[i=1:n])
#     @constraint(model, [i=1:n], d[i] == u[i] - x[i])
    
#     # Define objective function
#     @objective(model, Min, tau)

#     # Create variables w_j with enhanced bounds management
#     w = Dict()
        
#     # # Use adaptive bounds for w variables to prevent numerical issues
#     # w_bound = T(1e6) * max(one(T), maximum(abs, h))
#     # for j in 1:m
#     #     w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], 
#     #                    lower_bound=0, upper_bound=w_bound)
#     # end
    
#     for j in 1:m

#         w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)

#         if minimum(h) != 0.0
#             @constraint(model, dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], d) <= tau)
#         else
#             @constraint(model, dot(Jf[j,:], d) <= tau)
#         end

#         # Optimized constraint generation with sparsity awareness
#         A_j = SubProbParams.A[j]
#         for i in axes(A_j, 2)
#             # Build constraint expressions efficiently based on matrix sparsity
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
            
#             # Add the constraint with proper handling of degenerate cases
#             if iszero(expr)
#                 # If the expression is zero, u[i] must be zero
#                 @constraint(model, u[i] == 0.0)
#             else
#                 # Standard case: the linear combination of w variables equals u[i]
#                 @constraint(model, expr == u[i]) # A_j^T w_j = u
#             end
#         end
#     end
    
#     # Solve the optimization problem
#     optimize!(model)

#     # Enhanced solution status checking - focused on what really matters
#     status = termination_status(model)

#     if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
#         # Success case - extract solution
#         tau_val = value(tau)
#         u_val = [value(u[i]) for i in 1:n]
#         return tau_val, u_val, true, string(status)
        
#     elseif status == MOI.INFEASIBLE
#         # Infeasible case - this is mathematically important information
#         return convert(T, Inf), fill(convert(T, NaN), n), false, "Problem is infeasible"
        
#     else
#         # All other cases - treat as general failure
#         # This includes UNBOUNDED, TIME_LIMIT, ITERATION_LIMIT, and any other status
#         return convert(T, NaN), fill(convert(T, NaN), n), false, string(status)
#     end
# end

"""
    subproblem_condg(x, Jf, h, SubProbParams; tol = sqrt(eps(T)), max_iter = 10_000,
                     max_time = 5.0, verbose = 0, solver_cache = nothing, presolve = "on")

Resolve o subproblema linear do CondG:
    min τ
    s.a.  ⟨∇f_j(x), u - x⟩  + ( ⟨b_j, w_j⟩ - h_j ) ≤ τ,  se h_j ≉ 0
          ⟨∇f_j(x), u - x⟩                       ≤ τ,  se h_j ≈ 0
          A_j' w_j = u        (∀ j)
          lb ≤ u ≤ ub,  w_j ≥ 0

Retorna (tau::T, u::Vector{T}, success::Bool, message::String).
"""
function subproblem_condg(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T},
                              SubProbParams::SubProblemParams;
                              tol::T = sqrt(eps(T)),
                              max_iter::Int = 10_000,
                              max_time::Real = 5.0,
                              verbose::Int = 0,
                              solver_cache::Union{Nothing, Model} = nothing,
                              presolve::String = "on") where {T<:AbstractFloat}

    # checks leves
    if any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), length(x)), false, "Invalid input"
    end

    model = solver_cache === nothing ? Model(HiGHS.Optimizer) : (empty!(solver_cache); solver_cache)

    # verbosidade
    if verbose <= 2
        set_silent(model)
        set_optimizer_attribute(model, "output_flag", false)
        set_optimizer_attribute(model, "log_to_console", false)
    else
        unset_silent(model)
        set_optimizer_attribute(model, "output_flag", true)
        set_optimizer_attribute(model, "log_to_console", true)
    end

    # limites de recurso
    set_optimizer_attribute(model, "simplex_iteration_limit", max_iter)
    set_optimizer_attribute(model, "ipm_iteration_limit",      max_iter)
    set_optimizer_attribute(model, "time_limit",               float(max_time))
    # tolerâncias
    pf_tol = Float64(max(tol, T(1e-12)))
    df_tol = Float64(max(tol, T(1e-12)))
    set_optimizer_attribute(model, "primal_feasibility_tolerance", pf_tol)
    set_optimizer_attribute(model, "dual_feasibility_tolerance",   df_tol)
    set_optimizer_attribute(model, "presolve", presolve)

    # variáveis
    Iu = eachindex(SubProbParams.lb)  # índices consistentes com lb/ub/x
    @variable(model, tau)
    @variable(model, SubProbParams.lb[i] <= u[i in Iu] <= SubProbParams.ub[i])

    # w_j para cada j, dimensionados por linhas de A_j
    A_list = SubProbParams.A
    b_list = SubProbParams.b
    @objective(model, Min, tau)

    w = Vector{Vector{VariableRef}}(undef, length(A_list))
    for j in eachindex(A_list)
        Aj = A_list[j]
        # garantir que colunas de A_j casam com u
        @assert axes(Aj, 2) == Iu "Dimensão de A[$j] incompatível com u"
        rset = axes(Aj, 1)
        w[j] = @variable(model, [r in rset], lower_bound = 0.0)

        # A_j' w_j = u
        for i in axes(Aj, 2)
            @constraint(model, sum(Aj[r, i] * w[j][r] for r in rset) == u[i])
        end

        # ⟨∇f_j(x), u - x⟩ + (⟨b_j, w_j⟩ - h_j) ≤ τ  (se h_j ≉ 0)  /  ⟨∇f_j(x), u - x⟩ ≤ τ
        grad_j = view(Jf, j, axes(Jf,2))  # linha j de Jf
        if !isapprox(h[j], zero(T); atol = max(tol, T(1e-14)))
            @constraint(model, dot(grad_j, [u[i] - x[i] for i in Iu]) +
                                  (dot(b_list[j], w[j]) - h[j]) <= tau)
        else
            @constraint(model, dot(grad_j, [u[i] - x[i] for i in Iu]) <= tau)
        end
    end

    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        tau_val = convert(T, value(tau))
        u_val   = [convert(T, value(u[i])) for i in Iu]
        return tau_val, u_val, true, string(status)
    elseif status == MOI.INFEASIBLE
        return convert(T, Inf), [T(NaN) for _ in Iu], false, "Infeasible"
    else
        return convert(T, NaN), [T(NaN) for _ in Iu], false, string(status)
    end
end