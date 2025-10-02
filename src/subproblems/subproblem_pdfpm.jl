# function subproblem_pdfpm(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, 
#                            SubProbParams::SubProblemParams; 
#                            sigma::T=1.0, 
#                            B::Union{Nothing, Vector{<:AbstractMatrix{T}}}=nothing,
#                            tol::T=sqrt(eps(T)),
#                            max_iter::Int=3000,
#                            verbose::Int=0,
#                            warm_start::Union{Nothing, AbstractVector{T}}=nothing,
#                            solver_cache::Union{Nothing, Model}=nothing) where {T<:AbstractFloat}
    
#     n = length(x)
#     m = length(SubProbParams.A)
    
#     if !isfinite(sigma) || sigma < sqrt(eps(T)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid input parameters"
#     end
    
#     # Handle B matrices with additional checks
#     if B === nothing
#         B = [Matrix{T}(I, n, n) for _ in 1:m]
#     else
#         # Check for numerical issues in B
#         for (j, Bj) in enumerate(B)
#             if any(!isfinite, Bj)
#                 return convert(T, NaN), fill(convert(T, NaN), n), false, "Invalid B matrix at index $j"
#             end
#             # Check condition number for potential issues
#             cond_Bj = cond(Bj)
#             if cond_Bj > 1e12 && verbose > 0 # Very ill-conditioned
#                 @warn "B matrix $j has high condition number: $cond_Bj"
#             end
#         end
#     end

#     # Validate constraint matrices
#     for (j, Aj) in enumerate(SubProbParams.A)
#         # Duplicate check, maybe not needed
#         # if size(Aj, 2) != n
#         #     return convert(T, NaN), fill(convert(T, NaN), n), false, "Dimension mismatch in A[$j]"
#         # end
#         # Check for rank deficiency
#         if rank(Aj) < min(size(Aj)...)
#             @warn "Constraint matrix A[$j] may be rank deficient"
#         end
#     end

#     # Create or reuse model
#     if solver_cache === nothing
#         model = Model(Ipopt.Optimizer)
#     else
#         model = solver_cache
#         # Clear previous solution
#         empty!(model)
#     end
    
#     # Adaptive Ipopt settings based on precision
#     set_optimizer_attribute(model, "max_iter", max_iter)
#     set_optimizer_attribute(model, "tol", max(tol, 1e-14))  # Ipopt has minimum tolerance
#     set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
#     set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))
#     set_optimizer_attribute(model, "print_level", verbose)
    
#     # Numerical stability settings
#     set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
#     set_optimizer_attribute(model, "bound_relax_factor", max(tol/100, 1e-12))
#     set_optimizer_attribute(model, "honor_original_bounds", "yes")
    
#     # Additional robustness settings
#     set_optimizer_attribute(model, "mu_strategy", "adaptive")
#     set_optimizer_attribute(model, "jacobian_regularization_value", max(eps(T), 1e-8))
    
#     # Define variables with adaptive bounds
#     bound_scale = T(1e6) * max(one(T), maximum(abs, x))  # Scale bounds with problem
#     lb_safe = max.(SubProbParams.lb, -bound_scale)
#     ub_safe = min.(SubProbParams.ub, bound_scale)
    
#     # Check for feasibility
#     #QUESTION: if SubProbParams.lb > SubProbParams.ub, then lb_safe .> ub_safe?
#     if any(lb_safe .> ub_safe)
#         return convert(T, NaN), fill(convert(T, NaN), n), false, "Infeasible bounds"
#     end
    
#     @variable(model, tau)
#     @variable(model, lb_safe[i] <= u[i=1:n] <= ub_safe[i])
    
#     # Warm start if provided
#     # TODO: Add warm start to the main function
#     if warm_start !== nothing && length(warm_start) == n
#         for i in 1:n
#             if lb_safe[i] <= warm_start[i] <= ub_safe[i]
#                 set_start_value(u[i], warm_start[i])
#             end
#         end
#     end
    
#     # Use auxiliary variables for numerical stability
#     @variable(model, d[i=1:n])
#     @constraint(model, [i=1:n], d[i] == u[i] - x[i])
    
#     # Adaptive regularization based on sigma and precision
#     @objective(model, Min, tau + 0.5 * sigma * sum(d[i]^2 for i in 1:n))

#     # Create variables w_j with adaptive bounds
#     w = Dict()

#     # TODO: test if this is the best bound for w
#     # w_bound = T(1e4) * max(one(T), maximum(abs, h))
#     # for j in 1:m
#     #     num_constraints = size(SubProbParams.A[j], 1)
#     #     w[j] = @variable(model, [1:num_constraints], lower_bound=0, upper_bound=w_bound)
#     # end

#     # WARNING: Relies on Ipopt to handle scaling
#     for j in 1:m
#         w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)

#         if minimum(h) != 0.0
#             # Constraints with numerical safeguards
#             @constraint(model,
#                 dot(Jf[j,:], d) + 0.5 * dot(B[j] * d, d) + dot(SubProbParams.b[j], w[j]) - h[j] <= tau)
#         else
#             # Simplified constraints with the same numerical care
#             @constraint(model, dot(Jf[j,:], d) + 0.5 * dot(B[j] * d, d) <= tau)
#         end

#         # Linear constraints with sparsity awareness
#         A_j = SubProbParams.A[j]
#         for i in axes(A_j, 2)
#             # Build expression efficiently
#             expr = AffExpr()
            
#             if issparse(A_j)
#                 # For sparse matrices, iterate only over non-zeros
#                 for k in nzrange(A_j, i)
#                     row_idx = rowvals(A_j)[k]
#                     value = nonzeros(A_j)[k]
#                     expr += value * w[j][row_idx]
#                 end
#             else
#                 # For dense matrices, iterate over all elements
#                 for r in axes(A_j, 1)
#                     if !iszero(A_j[r, i])  # Skip zeros even in dense matrices
#                         expr += A_j[r, i] * w[j][r]
#                     end
#                 end
#             end
            
#             # Add the constraint
#             if iszero(expr)
#                 @constraint(model, u[i] == 0.0)
#             else
#                 @constraint(model, expr == u[i])
#             end
#         end
#     end
    
#     # Solve the problem
#     optimize!(model)

#     # Check solution status
#     status = termination_status(model)

#     if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
#         # Extract solution
#         tau_val = value(tau)
#         u_val = [value(u[i]) for i in 1:n]
#         theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
        
#         return theta, u_val, true, string(status)
#     else
#         # Return failure
#         return convert(T, NaN), fill(convert(T, NaN), n), false, string(status)
#     end
# end

"""
    subproblem_pdfpm(x, Jf, h, SubProbParams; sigma=1.0, B=nothing,
                       tol=sqrt(eps(T)), max_iter=3000, max_time=5.0,
                       verbose=0, warm_start=nothing, solver_cache=nothing) -> (θ, u, success, msg)

Resolve o subproblema quadrático (PDFPM) no ponto `x`:

    min τ + (σ/2)‖u - x‖²
    s.a.  ⟨∇f_j(x), u - x⟩  + (1/2)(u - x)' B_j (u - x)  + ( ⟨b_j, w_j⟩ - h_j ) ≤ τ,  se h_j ≉ 0
          ⟨∇f_j(x), u - x⟩  + (1/2)(u - x)' B_j (u - x)                      ≤ τ,  se h_j ≈ 0
          A_j' w_j = u   (∀ j)
          lb ≤ u ≤ ub,   w_j ≥ 0

Parâmetros:
- `sigma` (σ) controla a regularização quadrática isotrópica no objetivo.
- `B`: vetor de matrizes {B_j} opcionais (uma por objetivo). Se `nothing`, usa identidade.
- `tol`, `max_iter`, `max_time` e `verbose`: atributos do Ipopt.
- `warm_start`: chute inicial para `u` (respeitado se ∈ [lb,ub]).
- `solver_cache`: reuso de `Model` para chamadas repetidas.

Retorna:
- `θ = τ* + (σ/2)‖u* - x‖²` (valor do objetivo modelado),
- `u` (solução),
- `success::Bool`,
- `msg::String` (status do solver).
"""
function subproblem_pdfpm(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T},
                            SubProbParams::SubProblemParams;
                            sigma::T = one(T),
                            B::Union{Nothing, Vector{<:AbstractMatrix{T}}} = nothing,
                            tol::T = sqrt(eps(T)),
                            max_iter::Int = 3000,
                            max_time::Real = 5.0,
                            verbose::Int = 0,
                            warm_start::Union{Nothing, AbstractVector{T}} = nothing,
                            solver_cache::Union{Nothing, Model} = nothing) where {T<:AbstractFloat}

    # --- checks on input parameters
    if !(isfinite(sigma) && sigma ≥ T(0)) || any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), length(x)), false, "Invalid input"
    end

    # Basic dimensions and shortcuts
    Iu   = eachindex(x)                  # indices of u, x, lb/ub
    m    = length(SubProbParams.A)       # number of objectives
    A    = SubProbParams.A
    b    = SubProbParams.b
    lb   = SubProbParams.lb
    ub   = SubProbParams.ub

    # Handle B: if nothing, use identity for each j
    local B_loc
    if B === nothing
        B_loc = [I for _ in 1:m]
    else
        B_loc = B
    end
    @assert length(B_loc) == m "B must have same length as A/b/h/Jf"

    # ------------------------------------------------------------------
    # Build Ipopt model
    # ------------------------------------------------------------------
    model = solver_cache === nothing ? Model(Ipopt.Optimizer) : (empty!(solver_cache); solver_cache)

    # Verbosity and solver attributes
    if verbose > 0
        unset_silent(model)
    else
        set_silent(model)
    end
    set_optimizer_attribute(model, "print_level", verbose)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "max_cpu_time", float(max_time))
    set_optimizer_attribute(model, "tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))
    set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "honor_original_bounds", "yes")

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    @variable(model, tau)
    @variable(model, lb[i] <= u[i in Iu] <= ub[i])

    # Auxiliary direction d = u - x
    @variable(model, d[i in Iu])
    @constraint(model, [i in Iu], d[i] == u[i] - x[i])

    # Warm start: only if feasible
    if warm_start !== nothing && length(warm_start) == length(Iu)
        for i in Iu
            v = warm_start[i]
            if lb[i] <= v <= ub[i]
                set_start_value(u[i], v)
            end
        end
    end

    # ------------------------------------------------------------------
    # Objective
    # (Important: quadratic terms with B_j do NOT enter the objective
    # when h_j ≠ 0, they stay in the constraints. Only sigma term is in obj.)
    # ------------------------------------------------------------------
    @objective(model, Min, tau + 0.5*sigma*sum(d[i]^2 for i in Iu))

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    all_h_zero = all(j -> isapprox(h[j], zero(T); atol = max(tol, T(1e-14))), 1:m)

    if all_h_zero
        # Special case: h ≈ 0 for all j → no w_j, no A_j' w_j = u
        for j in 1:m
            grad_j = view(Jf, j, axes(Jf,2))
            quad_j = 0.5 * dot(B_loc[j] * d, d)
            @constraint(model, dot(grad_j, [d[i] for i in Iu]) + quad_j <= tau)
        end
    else
        # General case: need w_j and A_j' w_j = u for all j
        for j in 1:m
            Aj = A[j]
            @assert axes(Aj, 2) == Iu "A[$j] has wrong dimension"
            rset = axes(Aj, 1)

            # Slack-like variables w_j ≥ 0
            wj = @variable(model, [r in rset], lower_bound = 0.0)

            # A_j' w_j = u
            for i in axes(Aj, 2)
                @constraint(model, sum(Aj[r, i]*wj[r] for r in rset) == u[i])
            end

            # Constraint depending on h_j
            grad_j = view(Jf, j, axes(Jf,2))
            quad_j = 0.5 * dot(B_loc[j] * d, d)
            if !isapprox(h[j], zero(T); atol = max(tol, T(1e-14)))
                @constraint(model, dot(grad_j, [d[i] for i in Iu]) + quad_j +
                                   (dot(b[j], wj) - h[j]) <= tau)
            else
                @constraint(model, dot(grad_j, [d[i] for i in Iu]) + quad_j <= tau)
            end
        end
    end

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        τ  = convert(T, value(tau))
        ū  = [convert(T, value(u[i])) for i in Iu]
        θ  = τ + 0.5*sigma*sum((ū[i]-x[i])^2 for i in Iu)
        return θ, ū, true, string(status)
    elseif status == MOI.INFEASIBLE
        return convert(T, Inf), [T(NaN) for _ in Iu], false, "Infeasible"
    else
        return convert(T, NaN), [T(NaN) for _ in Iu], false, string(status)
    end
end