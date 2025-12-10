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

### Implementation Details

The evaluation of this structure is handled in `src/utils/evalh.jl`.
- **`evalh!`**: Computes $h(x)$ by solving the maximization problem over the set $Z_j$ defined above.

## Available Solvers

1.  **CondG (Conditional Gradient)**: Suitable when the domain $\Omega$ is compact. It avoids full projection steps by solving linear subproblems.
2.  **ProxGrad (Proximal Gradient)**: generalization of the classic gradient descent, using the proximal operator to handle the non-smooth term.
3.  **PDFPM (Partially Derivative-Free Proximal Method)**: A variation that estimates gradients via finite differences, useful when $\nabla f(x)$ is expensive or unavailable.
