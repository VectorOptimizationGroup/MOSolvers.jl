module MOSolvers

    using NLPModels
    using LinearAlgebra
    using SparseArrays
    using JuMP, HiGHS, Ipopt
    using Printf

    # Incluir funções auxiliares
    include("utils/solver_utils.jl")
    include("utils/linesearch.jl")

    # Incluir subproblemas
    include("subproblems/subproblem_condg.jl")
    include("subproblems/subproblem_proxgrad.jl")
    include("subproblems/subproblem_pdfpm.jl")

    # Incluir solvers
    include("solvers/condg.jl")
    include("solvers/proxgrad.jl")
    include("solvers/pdfpm.jl")

    # Exportar solvers
    export CondG, CondG_options, ProxGrad, ProxGrad_options, PDFPM, PDFPM_options
end