"""
    _convert_solver_bounds(::Type{T}, bound, n, default_value, name)

Convert optional vector bounds to the solver element type `T`.

If `bound` is omitted, return a length-`n` vector filled with `default_value`.
If `bound` is provided, it must be an `AbstractVector` with length `n`; scalar
bounds are rejected.
"""
function _convert_solver_bounds(::Type{T}, bound, n::Int, default_value::T, name::Symbol) where {T<:AbstractFloat}
    if bound === nothing
        return fill(default_value, n)
    elseif bound isa AbstractVector
        length(bound) == n || throw(ArgumentError("`$name` must have length $n."))
        return Vector{T}(bound)
    else
        throw(ArgumentError("`$name` must be an AbstractVector or omitted. Scalar bounds are not supported."))
    end
end
