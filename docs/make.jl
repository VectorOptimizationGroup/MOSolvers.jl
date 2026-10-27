using Documenter
using DocumenterCitations
using MOSolvers

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(
    root = @__DIR__,
    sitename = "MOSolvers.jl",
    modules = [MOSolvers],
    plugins = [bib],
    checkdocs = :exports,
    doctest = true,
    # Ipopt prints this license banner once per process, independently of verbosity.
    doctestfilters = [r"\s*\*+\nThis program contains Ipopt,[\s\S]*?\n\*+\s*|$"],
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls=true,
        edit_link=nothing,
        repolink="https://github.com/VectorOptimizationGroup/MOSolvers.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => [
            "Installation" => "installation.md",
            "Quick start" => "quickstart.md",
        ],
        "Guides" => [
            "Problem formulation" => "guides/problems.md",
            "Configuration" => "guides/configuration.md",
            "Results and stopping" => "guides/results.md",
            "Line search" => "guides/linesearch.md",
        ],
        "Algorithms" => [
            "Overview" => "algorithms/index.md",
            "CondG" => "algorithms/condg.md",
            "ProxGrad" => "algorithms/proxgrad.md",
            "PDFPM" => "algorithms/pdfpm.md",
        ],
        "API reference" => "api.md",
        "References" => "references.md",
    ],
)
