# Teste do sistema de status do solver CondG
# Este arquivo demonstra como falhas no evalh! são propagadas para o solver

using JuMP
using HiGHS

# Incluir os arquivos necessários
include("src/utils/solver_utils.jl")
include("src/solvers/condg.jl")

# Função de teste que sempre falha para demonstrar o sistema de status
function test_evalf(x)
    return [1.0, 2.0]  # Retorna valores fixos para teste
end

function test_evalJf(x)
    return [1.0 0.0; 0.0 1.0]  # Matriz identidade 2x2
end

# Configuração que causará falha no evalh!
# Matriz B e vetor b que tornam o problema infactível
B = [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)]  # Matrizes identidade
b = [[-1.0, -1.0], [-1.0, -1.0]]  # Vetores que tornam o problema infactível

x0 = [0.0, 0.0]  # Ponto inicial
delta = 1.0

println("Testando o sistema de status do solver CondG...")
println("Configuração que causará falha no evalh!:")

# Chamar o solver
result = CondG(test_evalf, test_evalJf, B, delta, x0)

# Verificar o resultado
println("\nResultado da otimização:")
println("Sucesso: ", result.success)
println("Mensagem: ", result.message)
println("Avaliações de função: ", result.n_f_evals)
println("Avaliações de jacobiana: ", result.n_Jf_evals)

if !result.success
    println("\n✅ Teste PASSED: O sistema de status detectou corretamente a falha no evalh!")
else
    println("\n❌ Teste FAILED: O sistema de status não detectou a falha no evalh!")
end 