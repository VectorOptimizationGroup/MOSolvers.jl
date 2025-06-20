# types/mo_composite_model.jl
mutable struct MOCompositeModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    
    # Número de funções objetivo
    nobj::Int
    
    # Funções para avaliar cada objetivo (parte suave)
    f::Vector{Function}
    
    # Funções para avaliar gradientes de cada objetivo (parte suave)
    grad_f::Vector{Function}
    
    # Funções para avaliar parte não-suave
    g::Vector{Function}
    
    # Contadores
    counters::Counters
end

# Construtor principal - usuário fornece funções f, grad_f e g
function MOCompositeModel(
    f::Vector{Function},
    grad_f::Vector{Function},
    g::Vector{Function},
    x0::AbstractVector{T};
    name::String = "MOCompositeModel",
    lvar::AbstractVector{T} = fill(-Inf, length(x0)),
    uvar::AbstractVector{T} = fill(Inf, length(x0))
) where T <: AbstractFloat
    
    nvar = length(x0)
    nobj = length(f)
    
    # Verificar consistência
    @assert length(grad_f) == nobj "Número de funções gradiente deve ser igual ao número de funções objetivo"
    @assert length(g) == nobj "Número de funções g deve ser igual ao número de funções objetivo"
    
    meta = NLPModelMeta(
        nvar,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        name = name
    )
    
    return MOCompositeModel(meta, nobj, f, grad_f, g, Counters())
end

# Métodos para avaliar componentes separados
function eval_f(nlp::MOCompositeModel, x::AbstractVector, i::Int)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)
    return nlp.f[i](x)
end

function eval_grad_f(nlp::MOCompositeModel, x::AbstractVector, i::Int)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_grad)
    return nlp.grad_f[i](x)
end

function eval_g(nlp::MOCompositeModel, x::AbstractVector, i::Int)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)
    return nlp.g[i](x)
end

# Métodos para avaliar a função completa (f + g)
function eval_obj(nlp::MOCompositeModel, x::AbstractVector, i::Int)
    return eval_f(nlp, x, i) + eval_g(nlp, x, i)
end

function eval_all_objectives(nlp::MOCompositeModel, x::AbstractVector)
    return [eval_obj(nlp, x, i) for i in 1:nlp.nobj]
end

# Implementações da API NLPModels
function NLPModels.obj(nlp::MOCompositeModel, x::AbstractVector, i::Int)
    return eval_obj(nlp, x, i)
end

function NLPModels.grad!(nlp::MOCompositeModel, x::AbstractVector, g::AbstractVector, i::Int)
    # Nota: Isso só calcula o gradiente da parte suave f
    # A parte não-suave g não tem gradiente definido aqui
    @lencheck nlp.meta.nvar x g
    g .= nlp.grad_f[i](x)
    return g
end