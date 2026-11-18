const _CONVERGENCE_STATUSES = (
    :optimality_tolerance,
    :proximal_step_tolerance,
)

"""
    converged(result)

Return whether a solver's stopping criterion was satisfied. This is derived
from `result.status`, not stored independently, and does not certify a global
optimum or equate the model measures of different algorithms.

Convergence statuses are CondG/ProxGrad's `:optimality_tolerance`, meaning
`|θ| < opt_tol` (including initial convergence and the explicit exit at `θ = 0`),
and PDFPM's `:proximal_step_tolerance`, meaning
`sigma * norm(p - x) < epsilon / (3(1 + maxⱼ tr(Bⱼ)))`; there `abs_theta` is `NaN`
because it was not recomputed at the returned point.

Other final statuses are `:max_iter`, `:max_time`, `:callback`,
`:max_subproblem_iter`, `:evaluation_failure`, `:subproblem_failure`,
`:line_search_failure`, and `:numerical_failure`. Details of component failures
are in `message`. Internal `:not_started` and `:running` states are not converged.
"""
converged(result::OptimResult) = result.status in _CONVERGENCE_STATUSES

"""Record one external termination cause and emit its summary."""
function _finalize_optimization!(result::OptimResult, options, total_start_time,
                                 status::Symbol, message::AbstractString)
    result.status = status
    result.message = String(message)
    result.total_time = time() - total_start_time
    if options.verbose > 0
        @info result.message status=result.status iter=result.iter total_time=result.total_time
    end
    return result
end
