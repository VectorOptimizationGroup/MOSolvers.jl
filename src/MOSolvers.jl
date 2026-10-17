"""
    MOSolvers

A package for Multi-Objective Optimization using descent methods.

# Solvers
- [`CondG`]: Conditional Gradient Method.
- [`ProxGrad`]: Proximal Gradient Method.
- [`PDFPM`]: Partially Derivative-Free Proximal Method.

# Configuration
Each solver has a corresponding options constructor:
- [`CondG_options`]
- [`ProxGrad_options`]
- [`PDFPM_options`]

`CondG` and `ProxGrad` take their line search from MultiObjectiveLineSearches;
`BackTracking` and `InitialStatic` are reexported.
"""
module MOSolvers

    using NLPModels
    using LinearAlgebra
    using SparseArrays
    using JuMP, HiGHS, Ipopt
    using Printf
    import MultiObjectiveLineSearches
    using MultiObjectiveLineSearches: BackTracking, InitialStatic

    # Include auxiliary functions
    include("utils/solver_utils.jl")
    include("internals/termination.jl")
    include("utils/input_conversion.jl")
    include("internals/linesearch_interface.jl")
    include("utils/evalh.jl")

    # Include subproblems
    include("subproblems/subproblem_condg.jl")
    include("subproblems/subproblem_proxgrad.jl")
    include("subproblems/subproblem_pdfpm.jl")

    # Include solvers
    include("solvers/condg.jl")
    include("solvers/proxgrad.jl")
    include("solvers/pdfpm.jl")

    include("internals/display.jl")

    # Export solvers
    export CondG, CondG_options, ProxGrad, ProxGrad_options, PDFPM, PDFPM_options
    export converged

    # Reexport the line-search configuration used by CondG and ProxGrad
    export BackTracking, InitialStatic

    # Export support-function utilities
    export SubProblemParams, evalh!, clear_evalh_cache!, set_evalh_highs_limits!
end
