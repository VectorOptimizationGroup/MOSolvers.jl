# MOSolvers.jl

Julia solvers for composite multiobjective optimization: [CondG](https://doi.org/10.1080/02331934.2023.2257709),
[ProxGrad](https://doi.org/10.1007/s10589-018-0043-x), and
[PDFPM](https://doi.org/10.48550/arXiv.2508.20071).

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/VectorOptimizationGroup/MultiObjectiveLineSearches.jl")
Pkg.add(url="https://github.com/VectorOptimizationGroup/MOSolvers.jl")
```

## Example

```julia
using MOSolvers

f(x) = [x[1]^2, (x[1] - 2)^2]
Jf(x) = reshape([2x[1], 2(x[1] - 2)], 2, 1)
A = [ones(1, 1), ones(1, 1)]

result = ProxGrad(f, Jf, A, 0.0, [3.0], ProxGrad_options(verbose=0);
                  lb=[-3.0], ub=[3.0])

converged(result)
result.x, result.Fval, result.status
```

## Documentation

The [documentation](docs/src/index.md) covers the problem formulation, solver
options, stopping criteria, line searches, API, and references.
See [docs/README.md](docs/README.md) to browse it on a local server.
