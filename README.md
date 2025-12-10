# MOSolvers.jl

**MOSolvers.jl** is a Julia package that provides the following implementations of algorithms for Multi-Objective Optimization (MO):
- **CondG**: [doi.org/10.1080/02331934.2023.2257709](https://doi.org/10.1080/02331934.2023.2257709)
- **ProxGrad**: [doi.org/10.1007/s10589-018-0043-x](https://doi.org/10.1007/s10589-018-0043-x)
- **PDFPM**: [arXiv:2508.20071](https://doi.org/10.48550/arXiv.2508.20071).

This package was originally developed to support the research presented in the paper cited below.

## Reference

If you use this package in your research, please cite the following paper:

- **PDFPM**: [arXiv:2508.20071](https://doi.org/10.48550/arXiv.2508.20071)

## Installation

This package is currently not registered in the General Julia registry. You can install it directly from GitHub using the package manager mode (press `]`) or `Pkg`:

```julia
import Pkg
Pkg.add(url="https://github.com/VectorOptimizationGroup/MOSolvers.jl")
```

## Usage Example

Here is a basic example of how to solve a multi-objective problem (BK1) using the Partially Derivative-Free Proximal Method (PDFPM) solver.

```julia
using MOSolvers
using LinearAlgebra

# 1. Define the Problem Data (BK1)
function bk1_evalf(x)
    return [x[1]^2 + x[2]^2, (x[1] - 5.0)^2 + (x[2] - 5.0)^2]
end

function bk1_evalJf(x)
    J = zeros(Float64, 2, 2)
    J[1, 1] = 2.0 * x[1]
    J[1, 2] = 2.0 * x[2]
    J[2, 1] = 2.0 * (x[1] - 5.0)
    J[2, 2] = 2.0 * (x[2] - 5.0)
    return J
end

# Initial guess and bounds
x0 = [5.0, 10.0]
lb = [-5.0, -5.0]
ub = [10.0, 10.0]

# Problem parameters for PDFPM (Partially Derivative-Free Proximal Method)
# A and delta correspond to the linearization model parameters.
# For standard problems, A_j is typically Identity and delta is 0.
A = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
delta = 0.0

# 2. Configure Solver Options
options = PDFPM_options(
    verbose=1,
    max_iter=100,
    opt_tol=1e-6,
    sigma=1.0,
    alpha=0.1
)

# 3. Solve the Problem
# Note: evalJf is optional for PDFPM, but can be provided if available.
result = PDFPM(bk1_evalf, A, delta, x0, options; evalJf=bk1_evalJf, lb=lb, ub=ub)

# 4. Inspect Results
println("Success: ", result.success)
println("Final x: ", result.x)
println("Final Objectives: ", result.Fval)
println("Iterations: ", result.iter)
```

## Package Structure

The source code is organized as follows:

- **`src/solvers/`**: Contains the main solver implementations.
  - `condg.jl`: Conditional Gradient Method.
  - `proxgrad.jl`: Proximal Gradient Method.
  - `pdfpm.jl`: Partially Derivative-Free Proximal Method.

- **`src/subproblems/`**: Solvers for the auxiliary subproblems required by each method.
  - `subproblem_condg.jl`
  - `subproblem_proxgrad.jl`
  - `subproblem_pdfpm.jl`

- **`src/utils/`**: Utility modules.
  - `linesearch.jl`: Multi-objective line search strategies (e.g., Armijo).
  - `evalh.jl`: Routines for evaluating supporting functions.
  - `solver_utils.jl`: Data structures like `OptimResult` and helper functions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements or report bugs.