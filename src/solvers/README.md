# Solvers and Problem Structure

The algorithms in **MOSolvers.jl** are designed to solve **Multiobjective Composite Optimization Problems** of the form:

$$
\min_{x \in \Omega} F(x) = f(x) + h(x)
$$

where:
- $f: \mathbb{R}^n \to \mathbb{R}^m$ is a continuously differentiable vector-valued function.
- $h: \mathbb{R}^n \to \mathbb{R}^m$ is a convex, coordinate-wise function, typically non-smooth.
- $\Omega \subset \mathbb{R}^n$ is a closed convex set (e.g., box constraints).

## Structure of the Non-Smooth Term $h(x)$

The solvers in this package are optimized for a specific case. We assume that each component $h_j(x)$ of the function $h(x)$ is defined as the maximum of a linear product over a compact set:

$$
h_j(x) = \max \{ \langle x, z \rangle \mid z \in Z_j \}
$$

In the current implementation, the set $Z_j$ is defined by:
 
$$
Z_j = \{ z \in \mathbb{R}^n \mid -\delta e \leq A_j z \leq \delta e \}.
$$

### User Inputs
The user must provide:
- **$m$ matrices** $A_j \in \mathbb{R}^{n \times n}$ (one for each objective).
- **A scalar** $\delta \in [0, 1]$.

> [!NOTE]
> **Special Case: $\delta = 0$**
> When $\delta = 0$, the non-smooth term $h(x)$ becomes zero (since $Z_j = \{0\}$), and the solver adapts to minimize the smooth function $F(x) = f(x)$.
> Even in this case, the user **must still provide** the $m$ matrices $A_j$ (e.g., identity matrices). The solver logic requires these inputs to define the problem structure, even if the resulting set $Z_j$ is a singleton.

### Implementation Details

The evaluation of this structure is handled in `src/utils/evalh.jl`.
- **`evalh!`**: Computes $h(x)$ by solving the maximization problem over the set $Z_j$ defined above.

## Available Solvers

1.  **CondG (Conditional Gradient)**: [doi.org/10.1080/02331934.2023.2257709](https://doi.org/10.1080/02331934.2023.2257709)
2.  **ProxGrad (Proximal Gradient)**: [doi.org/10.1007/s10589-018-0043-x](https://doi.org/10.1007/s10589-018-0043-x)
3.  **PDFPM (Partially Derivative-Free Proximal Method)**: [arXiv:2508.20071](https://doi.org/10.48550/arXiv.2508.20071).
