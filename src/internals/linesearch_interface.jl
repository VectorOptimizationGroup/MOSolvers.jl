# Line search of CondG and ProxGrad, built on MultiObjectiveLineSearches.

"""
    configure_linesearch(linesearch, F!, ::Type{T}, n, m) where {T}

Prepare `linesearch` for a CondG or ProxGrad run with element type `T`, `n`
variables and `m` objectives, where `F!(Fy, y)` fills `Fy` with the objective
values at `y` and returns `nothing`, or returns a message on failure. Nothing is
evaluated. These solvers search along `d = p - x`, so steps must not exceed one:
a larger `maxstep` or initial step raises `ArgumentError`, as do invalid options.
"""
function configure_linesearch(linesearch::BackTracking, F!, ::Type{T}, n::Integer, m::Integer) where {T<:AbstractFloat}
    linesearch.maxstep <= 1 && linesearch.initial_step.alpha <= 1 ||
        throw(ArgumentError("CondG and ProxGrad search along p - x: the `maxstep` and initial step of `linesearch` must not exceed 1."))
    # Finite prototypes fix the element type and the axes; their values are not used.
    return MultiObjectiveLineSearches.configure_linesearch(linesearch, F!, zeros(T, n), zeros(T, m))
end

function configure_linesearch(linesearch, F!, ::Type{T}, n::Integer, m::Integer) where {T<:AbstractFloat}
    throw(ArgumentError("Unsupported line-search specification of type $(typeof(linesearch))."))
end

"""
    perform_linesearch!(ls, x, JF, d, theta, Fx, slopes)

Search along `d = p - x` from `x`, where `Fx` holds the objective values at `x`.
The slopes `JF*d` are written to `slopes`, and the decrease factor is
`-abs(theta)`, so `theta` must be finite and nonzero.

Returns the `LineSearchResult` of MultiObjectiveLineSearches. On success, its
`x_trial` and `F_trial` hold the accepted point and objective values; they are
buffers of `ls`, so copy them before the next search.
"""
function perform_linesearch!(ls, x, JF, d, theta, Fx, slopes)
    mul!(slopes, JF, d)
    return MultiObjectiveLineSearches.perform_linesearch!(ls, x, d, Fx; slopes, factor=-abs(theta))
end
