function subproblem_condg(x::AbstractVector{T}, Jf::AbstractMatrix{T}, h::AbstractVector{T}, SubProbParams::SubProblemParams) where {T<:AbstractFloat}

    n = length(x)
    m = length(SubProbParams.A)

    # Create the model
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Define variables
    @variable(model, tau)
    @variable(model, SubProbParams.lb[i] <= u[i=1:n] <= SubProbParams.ub[i])  # Variável u com limites

    if minimum(h) != 0.0
        # Create variables w_j for each objective function
        w = Dict()
        for j in 1:m
            w[j] = @variable(model, [1:size(SubProbParams.A[j], 1)], lower_bound=0)
        end
    end

    # Define objective function
    @objective(model, Min, tau)

    if minimum(h) != 0.0
        # Restrições
        for j in 1:m
            # Restrição: <b_j, w_j> - h_j(x) + <∇f_j(x), u - x> ≤ tau
            @constraint(model, 
                dot(SubProbParams.b[j], w[j]) - h[j] + dot(Jf[j,:], u .- x) <= tau)
            
            # Restrição: A_j^T w_j = u
            # TODO:Descobrir quanto é size(SubProbParams.A[j], 1)
            for i in 1:n
                @constraint(model, 
                    sum(SubProbParams.A[j][r, i] * w[j][r] for r in 1:size(SubProbParams.A[j], 1)) == u[i])
            end
        end
    else
        # Restrições
        for j in 1:m
            # Restrição: <∇f_j(x), u - x> ≤ tau
            @constraint(model, 
                dot(Jf[j,:], u .- x) <= tau)
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
        message = "Linear program was solved!"
        tau_val = value(tau)
        u_val = [value(u[i]) for i in 1:n]
    
     else
        @warn "ERROR: Linear program could not be solved! Status: $(termination_status(model))"
        tau_val = convert(T, NaN)
        u_val = fill(convert(T, NaN), n)
        success = false
        message = "Linear program could not be solved! Status: $(termination_status(model))"
     end

    return tau_val, u_val, success, message
end