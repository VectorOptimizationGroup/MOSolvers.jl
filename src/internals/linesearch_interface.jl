# Line search of CondG and ProxGrad, built on MultiObjectiveLineSearches.

"""
    configure_linesearch(linesearch, ::Type{T}) where {T}

Configure `linesearch` for a CondG or ProxGrad run with element type `T`.
These solvers search along `d = p - x`, so steps must not exceed one: a larger
`maxstep` or initial step raises `ArgumentError`, as do invalid options.
"""
function configure_linesearch(linesearch::BackTracking, ::Type{T}) where {T<:AbstractFloat}
    linesearch.maxstep <= 1 && linesearch.initial_step.alpha <= 1 ||
        throw(ArgumentError("CondG and ProxGrad search along p - x: the `maxstep` and initial step of `linesearch` must not exceed 1."))
    return MultiObjectiveLineSearches.configure_linesearch(linesearch, T)
end

function configure_linesearch(linesearch, ::Type{T}) where {T<:AbstractFloat}
    throw(ArgumentError("Unsupported line-search specification of type $(typeof(linesearch))."))
end

"""
    perform_linesearch!(ls, F!, x, JF, d, theta, Fx, x_trial, F_trial, slopes)

Search along `d = p - x` from `x`, where `Fx` holds the objective values at `x`
and `F!(Fy, y)` fills `Fy` with those at `y` and returns `nothing`, or returns
a message on failure. The slopes `JF*d` are written to `slopes`, and the
decrease factor is `-abs(theta)`, so `theta` must be finite and nonzero.

Returns the `LineSearchResult` of MultiObjectiveLineSearches. On success,
`x_trial` and `F_trial` hold the accepted point and objective values.
"""
function perform_linesearch!(ls, F!, x, JF, d, theta, Fx, x_trial, F_trial, slopes)
    # Row by row, as armijo computes them, so that both searches make the same trials.
    for i in eachindex(slopes)
        slopes[i] = dot(view(JF, i, :), d)
    end
    return MultiObjectiveLineSearches.perform_linesearch!(ls, F!, x, d, Fx, slopes,
                                                         x_trial, F_trial; factor=-abs(theta))
end
