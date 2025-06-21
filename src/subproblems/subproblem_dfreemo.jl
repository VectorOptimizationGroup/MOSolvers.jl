function subproblem_dfreemo(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, SubProbParams::SubProblemParams; sigma::T=1.0, B::Union{Nothing, Vector{<:AbstractMatrix{T}}}=nothing) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)

    if B === nothing
        B = [Matrix{T}(I, n, n) for _ in 1:m]
    end

    # Create the model
    model = Model(Ipopt.Optimizer)
    
    # Configurações específicas para o Ipopt para melhorar a convergência
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-4)
    set_optimizer_attribute(model, "print_level", 0)
    
    # Define variables
    @variable(model, tau)
    @variable(model, SubProbParams.lb[i] <= u[i=1:n] <= SubProbParams.ub[i])  # Variável u com limites

    # Create variables w_j for each objective function
    w = Dict()
    for j in 1:m
        w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)
    end
    
    # Define objective function: tau + (sigma/2)||u - x||^2
    @objective(model, Min, tau + 0.5 * sigma * sum((u[i] - x[i])^2 for i in 1:n))

    # Restrições
    for j in 1:m
        # Restrição: <∇f_j, u - x> + 0.5 * <B_j * (u - x), (u - x)> + <b_j, w_j> - h_j(x) ≤ tau
        @constraint(model, 
            dot(Jf[j,:], u .- x) + 0.5 * dot(B[j] * (u .- x), (u .- x)) + dot(SubProbParams.b[j], w[j]) - h[j] <= tau)
        
        # Restrição: A_j^T w_j = u
        for i in 1:n
            @constraint(model, 
                sum(SubProbParams.A[j][r, i] * w[j][r] for r in 1:size(SubProbParams.A[j], 1)) == u[i])
        end
    end
    
    # Solve the problem
    optimize!(model)

    # Check if solution was found (accept multiple success statuses)
    if termination_status(model) == MOI.OPTIMAL || 
        termination_status(model) == MOI.LOCALLY_SOLVED || 
        termination_status(model) == MOI.ALMOST_OPTIMAL || 
        termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
        success = true
        message = "Nonlinear program was solved!"
        tau_val = value(tau)
        u_val = [value(u[i]) for i in 1:n]
        theta = tau_val + 0.5 * sigma * norm(u_val - x)^2
    
    else
        @warn "ERROR: Nonlinear program could not be solved! Status: $(termination_status(model))"
        tau_val = convert(T, NaN)
        u_val = fill(convert(T, NaN), n)
        theta = convert(T, NaN)
        success = false
        message = "Nonlinear program could not be solved! Status: $(termination_status(model))"
    end

    return theta, u_val, success, message
end
