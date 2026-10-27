# API reference

## Module

```@docs
MOSolvers
```

## Solvers

```@docs
MOSolvers.CondG
MOSolvers.ProxGrad
MOSolvers.PDFPM
```

## Options

```@docs
MOSolvers.CondG_options
MOSolvers.ProxGrad_options
MOSolvers.PDFPM_options
MOSolvers.CondGOptions
MOSolvers.ProxGradOptions
MOSolvers.PDFPMOptions
```

## Results

```@docs
MOSolvers.OptimResult
MOSolvers.converged
```

## Support-function utilities

```@docs
MOSolvers.SubProblemParams
MOSolvers.evalh!
MOSolvers.clear_evalh_cache!
MOSolvers.set_evalh_highs_limits!
```

## Line-search configuration

`BackTracking` and `InitialStatic` are provided by
[MultiObjectiveLineSearches.jl](https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl)
and exported by MOSolvers. See [Line search](@ref backtracking) for their use
with CondG and ProxGrad.
