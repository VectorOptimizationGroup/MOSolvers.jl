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

function subproblem_proxgrad(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, 
    SubProbParams::SubProblemParams; 
    sigma::T=1.0,
    tol::T=sqrt(eps(T)),
    max_iter::Int=3000,
    verbose::Int=0,
    warm_start::Union{Nothing, AbstractVector{T}}=nothing,
    solver_cache::Union{Nothing, Model}=nothing) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)

    # Enhanced input validation with detailed error checking
    if !isfinite(sigma) || sigma < sqrt(eps(T)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid input parameters"
    end

    if minimum(h) != 0.0
        # Validate constraint matrices with numerical stability checks
        for (j, Aj) in enumerate(SubProbParams.A)
            # Check for numerical issues in constraint matrices
            if any(!isfinite, Aj)
                return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid A matrix at index $j"
            end
            # Check for rank deficiency which could cause solver issues
            if rank(Aj) < min(size(Aj)...)
                @warn "Constraint matrix A[$j] may be rank deficient"
            end
        end
    end
    
    # Create or reuse model for better performance in repeated calls
    if solver_cache === nothing
        model = Model(Ipopt.Optimizer)
    else
        model = solver_cache
        # Clear previous solution to avoid contamination
        empty!(model)
    end

    # Adaptive Ipopt settings based on precision type and tolerance requirements
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", max(tol, 1e-14))  # Ipopt has minimum tolerance limits
    set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))  # More relaxed backup tolerance
    set_optimizer_attribute(model, "print_level", verbose)
    
    # Numerical stability settings that help with ill-conditioned problems
    set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "bound_relax_factor", max(tol/100, 1e-12))
    set_optimizer_attribute(model, "honor_original_bounds", "yes")
    
    # Additional robustness settings for difficult problems
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "jacobian_regularization_value", max(eps(T), 1e-8))
    
    # Define variables with adaptive bounds to prevent numerical overflow
    bound_scale = T(1e6) * max(one(T), maximum(abs, x))  # Scale bounds with problem magnitude
    lb_safe = max.(SubProbParams.lb, -bound_scale)
    ub_safe = min.(SubProbParams.ub, bound_scale)
    
    # Check for feasibility issues before proceeding
    if any(lb_safe .> ub_safe)
        return convert(T, NaN), fill(convert(T, NaN), n), false, "Infeasible bounds"
    end
    
    # Define variables
    @variable(model, tau)
    @variable(model, lb_safe[i] <= u[i=1:n] <= ub_safe[i])  # Use safe bounds
    
    # Warm start capability for better convergence when good initial guess available
    if warm_start !== nothing && length(warm_start) == n
        for i in 1:n
            if lb_safe[i] <= warm_start[i] <= ub_safe[i]
                set_start_value(u[i], warm_start[i])
            end
        end
    end
    
    # Use auxiliary variables for numerical stability in distance calculations
    @variable(model, d[i=1:n])
    @constraint(model, [i=1:n], d[i] == u[i] - x[i])

    # Define objective function with auxiliary variables for better conditioning
    @objective(model, Min, tau + 0.5 * sigma * sum(d[i]^2 for i in 1:n))
    
    if minimum(h) != 0.0
        # Create variables w_j with adaptive bounds for numerical stability
        w = Dict()

        # TODO: test if this is the best bound for w
        # w_bound = T(1e4) * max(one(T), maximum(abs, h))
        # for j in 1:m
        #     num_constraints = size(SubProbParams.A[j], 1)
        #     w[j] = @variable(model, [1:num_constraints], lower_bound=0, upper_bound=w_bound)
        # end
        
        # Use unbounded variables and let Ipopt handle scaling for robustness
        for j in 1:m
            w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)
        end

        for j in 1:m
            # Enhanced constraint formulation using auxiliary variables
            @constraint(model,
                dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], d) <= tau)
           
            # Optimized constraint generation with sparsity awareness for better performance
            A_j = SubProbParams.A[j]
            for i in axes(A_j, 2)
                # Build expression efficiently based on matrix sparsity
                expr = AffExpr()
                
                if issparse(A_j)
                    # For sparse matrices, iterate only over non-zero elements
                    for k in nzrange(A_j, i)
                        row_idx = rowvals(A_j)[k]
                        value = nonzeros(A_j)[k]
                        expr += value * w[j][row_idx]
                    end
                else
                    # For dense matrices, skip explicit zeros for efficiency
                    for r in axes(A_j, 1)
                        if !iszero(A_j[r, i])
                            expr += A_j[r, i] * w[j][r]
                        end
                    end
                end
                
                # Add the constraint with proper handling of zero expressions
                if iszero(expr)
                    @constraint(model, u[i] == 0.0)
                else
                    @constraint(model, expr == u[i])
                end
            end
        end

    else
        for j in 1:m
            # Simplified constraint using auxiliary variables for consistency
            @constraint(model, dot(Jf[j,:], d) <= tau)
        end  
    end
    
    # Solve the problem
    optimize!(model)

    # Enhanced solution status checking with more comprehensive success criteria
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        # Extract solution and compute objective value consistently
        tau_val = value(tau)
        u_val = [value(u[i]) for i in 1:n]
        theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
        
        return theta, u_val, true, string(status)
    else
        # Enhanced error handling with informative messages
        return convert(T, NaN), fill(convert(T, NaN), n), false, string(status)
    end
end