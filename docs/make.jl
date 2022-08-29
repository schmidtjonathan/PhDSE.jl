using PhDSE
using Documenter

DocMeta.setdocmeta!(PhDSE, :DocTestSetup, :(using PhDSE); recursive=true)

makedocs(;
    modules=[PhDSE],
    authors="Jonathan Schmidt <jonathan.schmidt@uni-tuebingen.de> and contributors",
    repo="https://github.com/schmidtjonathan/PhDSE.jl/blob/{commit}{path}#{line}",
    sitename="PhDSE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://schmidtjonathan.github.io/PhDSE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Kalman Filter" => "kalman_filter.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/schmidtjonathan/PhDSE.jl",
    devbranch="main",
)
