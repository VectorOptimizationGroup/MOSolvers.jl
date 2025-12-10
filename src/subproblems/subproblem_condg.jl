"""
    subproblem_condg(x, Jf, h, SubProbParams; kw...)

Solve the CondG linear subproblem at point `x`.

This function handles two cases:

**Special Case (all h ≈ 0):**
    min τ
    s.t.
    <∇f_j(x), u - x> ≤ τ,   (∀ j)
    lb ≤ u ≤ ub

**General Case (some h_j ≉ 0):**
    min τ
    s.t.
    <∇f_j(x), u - x> + (<b_j, w_j> - h_j) ≤ τ,  if h_j ≉ 0
    <∇f_j(x), u - x>                      ≤ τ,  if h_j ≈ 0
    A_j' w_j = u
    lb ≤ u ≤ ub,  w_j ≥ 0

# Arguments
- `x::AbstractVector`: Current point.
- `Jf::AbstractMatrix`: Jacobian of `f` at `x`.
- `h::AbstractVector`: Value of `h` at `x`.
- `SubProbParams::SubProblemParams`: Problem parameters (A, b, lb, ub).

# Keywords
- `tol::Real`: Tolerance for numerical comparisons and solver.
- `max_iter::Integer`: Maximum number of iterations.
- `max_time::Real`: Maximum execution time.
- `verbose::Integer`: Verbosity level.
- `solver_cache::Union{Nothing, Model}`: Reusable JuMP model.
- `presolve::String`: HiGHS presolve option ("on", "off", "auto").

# Returns
- `tau::T`: Optimal objective value.
- `u::Vector{T}`: Solution vector.
- `success::Bool`: Whether the solver converged.
- `message::String`: Status message.
"""
function subproblem_condg(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T},
                          SubProbParams::SubProblemParams;
                          tol::T = sqrt(eps(T)),
                          max_iter::Int = 10_000,
                          max_time::Real = 5.0,
                          verbose::Int = 0,
                          solver_cache::Union{Nothing, Model} = nothing,
                          presolve::String = "on") where {T<:AbstractFloat}

    # -----------------------
    # Lightweight input checks
    # -----------------------
    if any(!isfinite, x) || any(!isfinite, Jf) || any(!isfinite, h)
        return convert(T, NaN), fill(convert(T, NaN), length(x)), false, "Invalid input"
    end

    # Index set aligned with lb/ub/x
    Iu = eachindex(SubProbParams.lb)
    m  = length(h)

    # Detect special case: all h ≈ 0 (numerically)
    all_h_zero = all(j -> isapprox(h[j], zero(T); atol = max(tol, T(1e-14))), 1:m)

    # -----------------------
    # Build (or reuse) model
    # -----------------------
    model = solver_cache === nothing ? Model(HiGHS.Optimizer) : (empty!(solver_cache); solver_cache)

    # Verbosity
    if verbose <= 2
        set_silent(model)
        set_optimizer_attribute(model, "output_flag", false)
        set_optimizer_attribute(model, "log_to_console", false)
    else
        unset_silent(model)
        set_optimizer_attribute(model, "output_flag", true)
        set_optimizer_attribute(model, "log_to_console", true)
    end

    # Resource limits & tolerances (HiGHS uses Float64 internally)
    set_optimizer_attribute(model, "simplex_iteration_limit", max_iter)
    set_optimizer_attribute(model, "ipm_iteration_limit",      max_iter)
    set_optimizer_attribute(model, "time_limit",               float(max_time))
    pf_tol = Float64(max(tol, T(1e-12)))
    df_tol = Float64(max(tol, T(1e-12)))
    set_optimizer_attribute(model, "primal_feasibility_tolerance", pf_tol)
    set_optimizer_attribute(model, "dual_feasibility_tolerance",   df_tol)
    set_optimizer_attribute(model, "presolve", presolve)

    # -----------------------
    # Variables & objective
    # -----------------------
    @variable(model, tau)
    @variable(model, SubProbParams.lb[i] <= u[i in Iu] <= SubProbParams.ub[i])
    @objective(model, Min, tau)

    # -----------------------
    # Constraints
    # -----------------------
    if all_h_zero
        # Special model: only ⟨∇f_j(x), u - x⟩ ≤ τ for all j
        for j in 1:m
            grad_j = view(Jf, j, axes(Jf, 2))
            @constraint(model, dot(grad_j, [u[i] - x[i] for i in Iu]) <= tau)
        end
    else
        # General model: includes w_j and A_j' w_j = u for all j
        A_list = SubProbParams.A
        b_list = SubProbParams.b

        for j in 1:length(A_list)
            Aj = A_list[j]
            @assert axes(Aj, 2) == Iu "A[$j] has incompatible column axes with u"
            rset = axes(Aj, 1)

            # w_j ≥ 0
            wj = @variable(model, [r in rset], lower_bound = 0.0)

            # A_j' w_j = u
            for i in axes(Aj, 2)
                @constraint(model, sum(Aj[r, i] * wj[r] for r in rset) == u[i])
            end
            
            # ⟨∇f_j(x), u - x⟩ + (⟨b_j, w_j⟩ - h_j) ≤ τ
            grad_j = view(Jf, j, axes(Jf, 2))
            @constraint(model,dot(grad_j, [u[i] - x[i] for i in Iu]) + (dot(b_list[j], wj) - h[j]) <= tau)
        end
    end

    # -----------------------
    # Solve & status handling
    # -----------------------
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