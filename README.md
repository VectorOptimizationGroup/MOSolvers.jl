# MOSolvers.jl

Pacote Julia para solvers de otimização multiobjetivo (MO).

## Descrição

MOSolvers.jl é uma biblioteca que fornece implementações de algoritmos para resolver problemas de otimização multiobjetivo, incluindo:

- CondG: Método do Gradiente Condicional para MO
- ProxGrad: Método do Gradiente Proximal para MO
- PDFPM: Método Proximal Livre de Derivadas para MO

## Instalação

```julia
import Pkg
Pkg.add(url="https://github.com/seu-usuario/MOSolvers.jl")
```

## Uso Básico

```julia
using MOProblems
using MOSolvers

# Criar um problema ZDT1 com 30 variáveis
problem = ZDT1(30)

# Configurar o solver CondG
solver = CondG(problem)

# Executar o solver
result = solve!(solver)

# Acessar a solução
pareto_front = result.pareto_front
pareto_set = result.pareto_set
```

## Estrutura do Pacote

- `types/`: Definições de tipos
  - `mo_composite_model.jl`: Interface para problemas compostos (f+g)

- `solvers/`: Implementações de solvers
  - `condg.jl`: Solver CondG
  - `proxgrad.jl`: Solver ProxGrad
  - `pdfpm.jl`: Solver PDFPM

- `subproblems/`: Implementações de subproblemas
  - `subproblem_condg.jl`: Subproblema para CondG
  - `subproblem_proxgrad.jl`: Subproblema para ProxGrad
  - `subproblem_pdfpm.jl`: Subproblema para PDFPM

- `utils/`: Funções auxiliares
  - `armijo.jl`: Busca linear Armijo para MO
  - `solver_utils.jl`: Funções auxiliares para solvers

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças propostas ou envie um pull request. 