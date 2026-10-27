# MOSolvers.jl

MOSolvers.jl provides CondG, ProxGrad, and PDFPM for composite multiobjective
optimization with box constraints and a support-function nonsmooth term.

Start with [Installation](@ref) and [Quick start](@ref).

- [Problem formulation](@ref) describes objectives, matrices, and bounds.
- [Configuration](@ref) covers options, numeric types, callbacks, and traces.
- [Results and stopping](@ref) explains the returned values and termination causes.
- [Line search](@ref backtracking) covers step selection for CondG and ProxGrad.
- [Algorithms](@ref) compares the three methods and their derivative inputs.
- [API reference](@ref) lists types and functions.
- [References](@ref) lists the publications associated with the methods.
