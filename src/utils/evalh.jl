
# -----------------------------------------------------------------------------
# Lightweight cache for evalh! models to avoid rebuilding HiGHS models every call
# Keyed by the SubProblemParams instance (object identity). Each entry stores a
# vector with (model, z) for each objective j.
const _EVALH_MODEL_CACHE = IdDict{SubProblemParams, Vector{NamedTuple{(:model, :z), Tuple{Model, Vector{JuMP.VariableRef}}}}}()

# Optional resource limits for HiGHS inside evalh!
struct _HiGHSLimits
    time_limit::Union{Nothing, Float64}
    simplex_iteration_limit::Union{Nothing, Int}
    ipm_iteration_limit::Union{Nothing, Int}
    output_flag::Union{Nothing, Bool}
end

const _EVALH_LIMITS = Base.RefValue{_HiGHSLimits}(_HiGHSLimits(nothing, nothing, nothing, nothing))

"""
    set_evalh_highs_limits!(; time_limit=nothing, simplex_iteration_limit=nothing, ipm_iteration_limit=nothing, output_flag=nothing)

Configure resource limits for HiGHS models used by `evalh!`.

All arguments are optional. Pass `nothing` to keep unset.
"""
function set_evalh_highs_limits!(; time_limit::Union{Nothing, Real}=nothing,
                                    simplex_iteration_limit::Union{Nothing, Integer}=nothing,
                                    ipm_iteration_limit::Union{Nothing, Integer}=nothing,
                                    output_flag::Union{Nothing, Bool}=nothing)
    _EVALH_LIMITS[] = _HiGHSLimits(
        time_limit === nothing ? nothing : Float64(time_limit),
        simplex_iteration_limit === nothing ? nothing : Int(simplex_iteration_limit),
        ipm_iteration_limit === nothing ? nothing : Int(ipm_iteration_limit),
        output_flag,
    )
    return nothing
end

function _ensure_h_models!(SubProbParams::SubProblemParams{T}, n::Int) where {T<:AbstractFloat}
    m = length(SubProbParams.A)
    cached = get(_EVALH_MODEL_CACHE, SubProbParams, nothing)
    # If missing or dimension mismatch, (re)build
    if cached === nothing || length(cached) != m || length(cached[1].z) != n
        models = Vector{NamedTuple{(:model, :z), Tuple{Model, Vector{JuMP.VariableRef}}}}(undef, m)
        for j in 1:m
            model = Model(HiGHS.Optimizer)
            set_silent(model)
            # Apply current resource limits (if any)
            lim = _EVALH_LIMITS[]
            if lim.time_limit !== nothing
                set_optimizer_attribute(model, "time_limit", lim.time_limit)
            end
            if lim.simplex_iteration_limit !== nothing
                set_optimizer_attribute(model, "simplex_iteration_limit", lim.simplex_iteration_limit)
            end
            if lim.ipm_iteration_limit !== nothing
                set_optimizer_attribute(model, "ipm_iteration_limit", lim.ipm_iteration_limit)
            end
            if lim.output_flag !== nothing
                set_optimizer_attribute(model, "output_flag", lim.output_flag)
            end
            @variable(model, z[1:n])
            @constraint(model, SubProbParams.A[j] * z .<= SubProbParams.b[j])
            models[j] = (model=model, z=z)
        end
        _EVALH_MODEL_CACHE[SubProbParams] = models
        return models
    else
        return cached
    end
end

"""
    clear_evalh_cache!()

Clear the cached HiGHS models used by `evalh!`.
Useful to release memory between runs with different problems.
"""
function clear_evalh_cache!()
    empty!(_EVALH_MODEL_CACHE)
    return nothing
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
    
    # Build or reuse cached models
    models = _ensure_h_models!(SubProbParams, n)

    for j = 1:m
        mdl = models[j].model
        z   = models[j].z

        # Update objective: maximize dot(x, z)
        JuMP.set_objective(mdl, MOI.MAX_SENSE, dot(x, z))

        # Re-apply limits in case they were changed by the user after caching
        lim = _EVALH_LIMITS[]
        if lim.time_limit !== nothing
            set_optimizer_attribute(mdl, "time_limit", lim.time_limit)
        end
        if lim.simplex_iteration_limit !== nothing
            set_optimizer_attribute(mdl, "simplex_iteration_limit", lim.simplex_iteration_limit)
        end
        if lim.ipm_iteration_limit !== nothing
            set_optimizer_attribute(mdl, "ipm_iteration_limit", lim.ipm_iteration_limit)
        end
        if lim.output_flag !== nothing
            set_optimizer_attribute(mdl, "output_flag", lim.output_flag)
        end

        # Solve the problem
        optimize!(mdl)

        # Check if solution was found (accept multiple success statuses)
        ts = termination_status(mdl)
        if ts == MOI.OPTIMAL || ts == MOI.LOCALLY_SOLVED || ts == MOI.ALMOST_OPTIMAL || ts == MOI.ALMOST_LOCALLY_SOLVED
            h[j] = objective_value(mdl)
        else
            @warn "ERROR: Linear program could not be solved! Status: $(ts)"
            return false, "Failed evaluating h_$j(x^k)."
        end
    end

    return true, "All subproblems solved successfully"
end
