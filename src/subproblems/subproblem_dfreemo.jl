function subproblem_dfreemo(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, 
                           SubProbParams::SubProblemParams; 
                           sigma::T=1.0, 
                           B::Union{Nothing, Vector{<:AbstractMatrix{T}}}=nothing,
                           tol::T=sqrt(eps(T)),
                           max_iter::Int=3000,
                           verbose::Int=0,
                           warm_start::Union{Nothing, AbstractVector{T}}=nothing,
                           solver_cache::Union{Nothing, Model}=nothing) where {T<:AbstractFloat}
    
    n = length(x)
    m = length(SubProbParams.A)
    
    if !isfinite(sigma) || sigma < sqrt(eps(T)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid input parameters"
    end
    
    # Handle B matrices with additional checks
    if B === nothing
        B = [Matrix{T}(I, n, n) for _ in 1:m]
    else
        # Check for numerical issues in B
        for (j, Bj) in enumerate(B)
            if any(!isfinite, Bj)
                return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid B matrix at index $j"
            end
            # Check condition number for potential issues
            cond_Bj = cond(Bj)
            if cond_Bj > 1e12  # Very ill-conditioned
                @warn "B matrix $j has high condition number: $cond_Bj"
            end
        end
    end
    
    # Validate constraint matrices
    for (j, Aj) in enumerate(SubProbParams.A)
        # Duplicate check, maybe not needed
        # if size(Aj, 2) != n
        #     return convert(T, NaN), fill(convert(T, NaN), n), false, "Dimension mismatch in A[$j]"
        # end
        # Check for rank deficiency
        if rank(Aj) < min(size(Aj)...)
            @warn "Constraint matrix A[$j] may be rank deficient"
        end
    end
    
    # Create or reuse model
    if solver_cache === nothing
        model = Model(Ipopt.Optimizer)
    else
        model = solver_cache
        # Clear previous solution
        empty!(model)
    end
    
    # Adaptive Ipopt settings based on precision
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", max(tol, 1e-14))  # Ipopt has minimum tolerance
    set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))
    set_optimizer_attribute(model, "print_level", verbose)
    
    # Numerical stability settings
    set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "bound_relax_factor", max(tol/100, 1e-12))
    set_optimizer_attribute(model, "honor_original_bounds", "yes")
    
    # Additional robustness settings
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "jacobian_regularization_value", max(eps(T), 1e-8))
    
    # Define variables with adaptive bounds
    bound_scale = T(1e6) * max(one(T), maximum(abs, x))  # Scale bounds with problem
    lb_safe = max.(SubProbParams.lb, -bound_scale)
    ub_safe = min.(SubProbParams.ub, bound_scale)
    
    # Check for feasibility
    #QUESTION: if SubProbParams.lb > SubProbParams.ub, then lb_safe .> ub_safe?
    if any(lb_safe .> ub_safe)
        return convert(T, NaN), fill(convert(T, NaN), n), false, "Infeasible bounds"
    end
    
    @variable(model, tau)
    @variable(model, lb_safe[i] <= u[i=1:n] <= ub_safe[i])
    
    # Warm start if provided
    # TODO: Add warm start to the main function
    if warm_start !== nothing && length(warm_start) == n
        for i in 1:n
            if lb_safe[i] <= warm_start[i] <= ub_safe[i]
                set_start_value(u[i], warm_start[i])
            end
        end
    end
    
    # Use auxiliary variables for numerical stability
    @variable(model, d[i=1:n])
    @constraint(model, [i=1:n], d[i] == u[i] - x[i])
    
    # Adaptive regularization based on sigma and precision
    @objective(model, Min, tau + 0.5 * sigma * sum(d[i]^2 for i in 1:n))
    
    if minimum(h) != 0.0
        # Create variables w_j with adaptive bounds
        w = Dict()

        # TODO: test if this is the best bound for w
        # w_bound = T(1e4) * max(one(T), maximum(abs, h))
        # for j in 1:m
        #     num_constraints = size(SubProbParams.A[j], 1)
        #     w[j] = @variable(model, [1:num_constraints], lower_bound=0, upper_bound=w_bound)
        # end

        # WARNING: Relies on Ipopt to handle scaling
        for j in 1:m
            w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)
        end
        
        # Constraints with numerical safeguards
        for j in 1:m
            # Build the constraint expression directly
            # Note: dot(Bd, d) and dot(Jf[j,:], d) return JuMP expressions, not numbers
            @constraint(model,
                dot(Jf[j,:], d) + 0.5 * dot(B[j] * d, d) + dot(SubProbParams.b[j], w[j]) - h[j] <= tau)
            
            # Linear constraints with sparsity awareness
            A_j = SubProbParams.A[j]
            for i in axes(A_j, 2)
                # Build expression efficiently
                expr = AffExpr()
                
                if issparse(A_j)
                    # For sparse matrices, iterate only over non-zeros
                    for k in nzrange(A_j, i)
                        row_idx = rowvals(A_j)[k]
                        value = nonzeros(A_j)[k]
                        expr += value * w[j][row_idx]
                    end
                else
                    # For dense matrices, iterate over all elements
                    for r in axes(A_j, 1)
                        if !iszero(A_j[r, i])  # Skip zeros even in dense matrices
                            expr += A_j[r, i] * w[j][r]
                        end
                    end
                end
                
                # Add the constraint
                if iszero(expr)
                    @constraint(model, u[i] == 0.0)
                else
                    @constraint(model, expr == u[i])
                end
            end
        end
    else
        # Simplified constraints with the same numerical care
        for j in 1:m
            @constraint(model, dot(Jf[j,:], d) + 0.5 * dot(B[j] * d, d) <= tau)
        end
    end
    
    # Solve the problem
    optimize!(model)

    # Check solution status
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        # Extract solution
        tau_val = value(tau)
        u_val = [value(u[i]) for i in 1:n]
        theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
        
        return theta, u_val, true, string(status)
    else
        # Return failure
        return convert(T, NaN), fill(convert(T, NaN), n), false, string(status)
    end
end