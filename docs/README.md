# Local documentation

The sources are in `docs/src/`. Documenter.jl, DocumenterCitations.jl, and
LiveServer.jl run in a separate Julia environment.

With MOSolvers.jl and MultiObjectiveLineSearches.jl checked out side by side,
run this from the MOSolvers.jl repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(path="../MultiObjectiveLineSearches.jl")]); Pkg.instantiate()'
```

Build the documentation and run its doctests:

```sh
julia --project=docs docs/make.jl
```

Start the local server:

```sh
julia --project=docs -e 'using LiveServer; serve(dir="docs/build")'
```

Open <http://localhost:8000/> in a browser. Stop the server with `Ctrl+C`.

The site is generated in `docs/build/` with directory URLs suitable for GitHub
Pages.
