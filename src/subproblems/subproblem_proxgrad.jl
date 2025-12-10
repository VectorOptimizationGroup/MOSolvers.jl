"""
    subproblem_proxgrad(x, Jf, h, SubProbParams;
                        sigma=1.0, tol=sqrt(eps(T)),
                        max_iter=3000, max_time=5.0,
                        verbose=0, warm_start=nothing,
    subproblem_proxgrad(x, Jf, h, SubProbParams; kw...)

Solve the proximal subproblem at point `x`.

This function handles two cases:

**Special Case (all h ≈ 0):**
    min τ + (σ/2)||u - x||^2
    s.t.
    <∇f_j(x), u - x> ≤ τ,   (∀ j)
    lb ≤ u ≤ ub

**General Case (some h_j ≉ 0):**
    min τ + (σ/2)||u - x||^2
    s.t.
    <∇f_j(x), u - x> + (<b_j, w_j> - h_j) ≤ τ,   if h_j ≉ 0
    <∇f_j(x), u - x>                      ≤ τ,   if h_j ≈ 0
    A_j' w_j = u
    lb ≤ u ≤ ub,  w_j ≥ 0

# Arguments
- `x::AbstractVector`: Current point.
- `Jf::AbstractMatrix`: Jacobian of `f` at `x`.
- `h::AbstractVector`: Value of `h` at `x`.
- `SubProbParams::SubProblemParams`: Problem parameters.

# Keywords
- `sigma::Real`: Proximal parameter σ.
- `tol::Real`: Tolerance.
- `max_iter::Integer`: Maximum number of iterations.
- `max_time::Real`: Maximum execution time.
- `verbose::Integer`: Verbosity level.
- `warm_start::Union{Nothing, AbstractVector}`: Initial guess for `u`.
- `solver_cache::Union{Nothing, Model}`: Reusable JuMP model.

# Returns
- `theta::T`: Modeled objective value: `τ* + (σ/2)||u* - x||^2`.
- `u::Vector{T}`: Solution vector.
- `success::Bool`: Whether the solver converged.
- `message::String`: Status message.
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

    # -- light checks --
    if !(isfinite(sigma) && sigma ≥ T(0)) || any(!isfinite, x) ||
       any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), length(x)), false, "Invalid input"
    end

    # shortcuts and axes
    Iu = eachindex(x)
    m  = length(SubProbParams.A)
    A  = SubProbParams.A
    b  = SubProbParams.b
    lb = SubProbParams.lb
    ub = SubProbParams.ub

    # -- (Re)creates model --
    model = solver_cache === nothing ? Model(Ipopt.Optimizer) : (empty!(solver_cache); solver_cache)
    if verbose > 0; unset_silent(model); else; set_silent(model); end
    set_optimizer_attribute(model, "print_level", verbose)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "max_cpu_time", float(max_time))
    set_optimizer_attribute(model, "tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "constr_viol_tol", max(tol, 1e-14))
    set_optimizer_attribute(model, "acceptable_tol", max(10*tol, 1e-12))
    set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "honor_original_bounds", "yes")

    # -- variables --
    @variable(model, tau)
    @variable(model, lb[i] <= u[i in Iu] <= ub[i])

    # d = u - x (for the proximal term)
    @variable(model, d[i in Iu])
    @constraint(model, [i in Iu], d[i] == u[i] - x[i])

    # warm-start (if feasible)
    if warm_start !== nothing && length(warm_start) == length(Iu)
        for i in Iu
            v = warm_start[i]
            if lb[i] <= v <= ub[i]
                set_start_value(u[i], v)
            end
        end
    end

    # objective: τ + (σ/2)‖d‖²
    @objective(model, Min, tau + 0.5*sigma*sum(d[i]^2 for i in Iu))

    # -- branch h ≈ 0 (all components) --
    all_h_zero = all(j -> isapprox(h[j], zero(T); atol = max(tol, T(1e-14))), 1:m)

    if all_h_zero
        # ⟨∇f_j(x), d⟩ ≤ τ
        for j in 1:m
            grad_j = view(Jf, j, axes(Jf,2))
            @constraint(model, sum(grad_j[i] * d[i] for i in Iu) <= tau)
        end
    else
        # General case: with w_j and A_j' w_j = u for all j
        for j in 1:m
            Aj = A[j]
            @assert axes(Aj, 2) == Iu "Dimension of A[$j] incompatible with u"
            rset = axes(Aj, 1)

            # A_j' * w_j = u, w_j ≥ 0
            wj = @variable(model, [r in rset], lower_bound = 0.0)
            for i in axes(Aj, 2)
                @constraint(model, sum(Aj[r, i]*wj[r] for r in rset) == u[i])
            end

            # ⟨∇f_j(x), d⟩ + (⟨b_j, w_j⟩ - h_j) ≤ τ  (only if h_j ≉ 0)
            grad_j = view(Jf, j, axes(Jf,2))
            if !isapprox(h[j], zero(T); atol = max(tol, T(1e-14)))
                @constraint(model, sum(grad_j[i]*d[i] for i in Iu) + (dot(b[j], wj) - h[j]) <= tau)
            else
                @constraint(model, sum(grad_j[i]*d[i] for i in Iu) <= tau)
            end
        end
    end

    # -- solve --
    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        τ = convert(T, value(tau))
        ū = [convert(T, value(u[i])) for i in Iu]
        θ = τ + 0.5*sigma*sum((ū[i] - x[i])^2 for i in Iu)
        return θ, ū, true, string(status)
    elseif status == MOI.INFEASIBLE
        return convert(T, Inf), [T(NaN) for _ in Iu], false, "Infeasible"
    else
        return convert(T, NaN), [T(NaN) for _ in Iu], false, string(status)
    end
end