"""
    subproblem_pdfpm(x, Jf, h, SubProbParams; sigma=1.0, B=nothing,
                       tol=sqrt(eps(T)), max_iter=3000, max_time=5.0,
                       verbose=0, warm_start=nothing, solver_cache=nothing) -> (θ, u, success, msg)

Solves the quadratic subproblem (PDFPM) at point `x`:

    min τ + (σ/2)‖u - x‖²
    s.t.  ⟨∇f_j(x), u - x⟩  + (1/2)(u - x)' B_j (u - x)  + ( ⟨b_j, w_j⟩ - h_j ) ≤ τ,  if h_j ≉ 0
          ⟨∇f_j(x), u - x⟩  + (1/2)(u - x)' B_j (u - x)                      ≤ τ,  if h_j ≈ 0
          A_j' w_j = u   (∀ j)
          lb ≤ u ≤ ub,   w_j ≥ 0

Parameters:
- `sigma` (σ) controls the quadratic regularization in the objective.
- `B`: vector of optional {B_j} matrices (one per objective). If `nothing`, uses identity.
- `tol`, `max_iter`, `max_time` and `verbose`: Ipopt attributes.
- `warm_start`: initial guess for `u` (respected if ∈ [lb,ub]).
- `solver_cache`: reuse of `Model` for repeated calls.

Returns:
- `θ = τ* + (σ/2)‖u* - x‖²` (modeled objective value),
- `u` (solution),
- `success::Bool`,
- `msg::String` (solver status).
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