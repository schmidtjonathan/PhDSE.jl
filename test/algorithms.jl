using Test
using LinearAlgebra
using Random
using Distributions
using ForwardDiff

using PhDSE

const PLOT_RESULTS = false



function matrix_fraction_decomposition(drift_matrix, dispersion_matrix, dt)
    dim = size(drift_matrix, 1)

    Φ = [
        drift_matrix   dispersion_matrix * dispersion_matrix'
        zeros(size(drift_matrix))   -drift_matrix'
        ]

    M = exp(Φ .* dt)

    Ah = M[1:dim, 1:dim]
    Qh = M[1:dim, dim+1:end] * Ah'

    return Ah, Qh
end


function simulate_linear(
    A,
    Q,
    u,
    H,
    R,
    v,
    μ₀,
    Σ₀,
    N::Int;
    rng = Random.GLOBAL_RNG,
)
    x = rand(rng, MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(rng, MvNormal(A * states[end] .+ u, Q)))
        push!(observations, rand(rng, MvNormal(H * states[end] .+ v, R)))
    end
    return states, observations
end

stack(x) = copy(reduce(hcat, x)')

function filtering_setup(D=100, d=10, num_obs=200)

    μ₀ = rand(D)
    Σ₀ = diagm(0=>ones(D))

    tspan = (0.0, 10.0)
    dt = (tspan[2] - tspan[1]) / num_obs
    F = diagm(0=>-rand(D))
    B = Diagonal(ones(D))
    A, Q = matrix_fraction_decomposition(F, B, dt)

    H = Diagonal(ones(D))[1:(D ÷ d):end, :]
    R = Diagonal(ones(d))

    u = zeros(D)
    v = 0.1 .* ones(d)

    ground_truth, observations = simulate_linear(A, Q, u, H, R, v, μ₀, Σ₀, num_obs)

    return μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations
end

@info "Executing tests for standard Kalman filters"
include("algorithms/kalman.jl")
@info "Executing tests for square-root Kalman filters"
include("algorithms/sqrt_kalman.jl")
# @info "Executing tests for ensemble Kalman filters"
# include("algorithms/enkf.jl")
