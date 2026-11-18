# Fixed cases of the bit-for-bit regression against commit 73e6de0: PDFPM with the
# ε-trace decrease and BFGS, ProxGrad and CondG. Generated reference in
# regression_reference.jl; the generator ran this file at 73e6de0.
using LinearAlgebra

# Nonsingular Ãⱼ, the same for every δ.
regression_A(n, j) = Matrix(1.0I, n, n) .+ (0.1 * j / n^2) .* reshape(1.0:n^2, n, n)

function regression_problems()
    quad = (name=:quad, x0=[2.0, -1.5], lb=fill(-5.0, 2), ub=fill(5.0, 2),
            evalf=x -> [sum(abs2, x), sum(abs2, x .- 1)],
            evalJf=x -> vcat(permutedims(2x), permutedims(2(x .- 1))))
    bk1 = (name=:bk1, x0=[3.0, 7.0], lb=fill(-5.0, 2), ub=fill(10.0, 2),
           evalf=x -> [x[1]^2 + x[2]^2, (x[1] - 5)^2 + (x[2] - 5)^2],
           evalJf=x -> [2x[1] 2x[2]; 2(x[1] - 5) 2(x[2] - 5)])
    # ZDT2 with n = 4: f₁ = x₁, f₂ = g(1 - (x₁/g)²), g = 1 + 3(x₂ + x₃ + x₄).
    zdt2 = (name=:zdt2, x0=[0.3, 0.2, 0.6, 0.1], lb=zeros(4), ub=ones(4),
            evalf=x -> (g = 1 + 3sum(x[2:4]); [x[1], g * (1 - (x[1] / g)^2)]),
            evalJf=x -> begin
                g = 1 + 3sum(x[2:4])
                d = 3 * (1 + (x[1] / g)^2)
                [1.0 0 0 0; -2x[1]/g d d d]
            end)
    # Hölder gradient: f₂ = |x₁|^1.5 + |x₂|^1.5 + |x₃|^1.5.
    holder = (name=:holder, x0=[1.2, -0.7, 0.4], lb=fill(-2.0, 3), ub=fill(2.0, 3),
              evalf=x -> [0.5sum(abs2, x .- [1.0, 0.5, -0.5]), sum(t -> abs(t)^1.5, x)],
              evalJf=x -> vcat(permutedims(x .- [1.0, 0.5, -0.5]), permutedims(1.5 .* sign.(x) .* sqrt.(abs.(x)))))
    # Large curvature: the first model steps overshoot and σ doubles.
    stiff = (name=:stiff, x0=[3.0, 2.0], lb=fill(-5.0, 2), ub=fill(5.0, 2),
             evalf=x -> [25sum(abs2, x), 25((x[1] - 1)^2 + x[2]^2)],
             evalJf=x -> [50x[1] 50x[2]; 50(x[1] - 1) 50x[2]])
    return (quad, bk1, zdt2, holder, stiff)
end

const REGRESSION_DELTAS = (0.0, 0.05)

# The fields compared; floating-point vectors as their bits.
regression_record(r) = (status=r.status, iter=r.iter, n_f_evals=r.n_f_evals, n_Jf_evals=r.n_Jf_evals,
                        n_sigma_doublings=r.n_sigma_doublings,
                        x=reinterpret(UInt64, collect(r.x)), Fval=reinterpret(UInt64, collect(r.Fval)))
