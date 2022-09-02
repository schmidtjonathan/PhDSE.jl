ENV["GKSwstype"] = 100

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
        "Algorithms" => [
            "Kalman Filter" => "algorithms/kalman_filter.md",
            "Square-root Kalman filter" => "algorithms/sqrt_kalman_filter.md"
        ],
        "Examples" => [
            "Kalman Filter" => "examples/kalman_filter.md",
            "Square-root Kalman Filter" => "examples/sqrt_kalman_filter.md",
            "Advanced: Solve PDE" => "examples/solve_1d_heat_eq.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/schmidtjonathan/PhDSE.jl",
    devbranch="main",
)
