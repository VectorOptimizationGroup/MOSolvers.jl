# Installation

Use Julia 1.11 or later. Install the packages into the active Julia environment:

```julia
import Pkg
Pkg.add(url="https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl")
Pkg.add(url="https://github.com/VectorOptimizationGroup/MOSolvers.jl")
```

For a local checkout, install the line-search dependency and develop MOSolvers:

```julia
import Pkg
Pkg.add(url="https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl")
Pkg.develop(path="/absolute/path/to/MOSolvers.jl")
```

The installation commands modify the active environment. The local path must
point to your checkout.

Verify that the package loads:

```jldoctest installation
julia> using MOSolvers

julia> options = ProxGrad_options(verbose=0);

julia> options.max_iter
1000
```

Continue with [Quick start](@ref) for a complete solve.
