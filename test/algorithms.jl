using Test
using LinearAlgebra
using StatsBase
using Statistics
using Random
using Distributions
using ForwardDiff

using PhDSE

const PLOT_RESULTS = false

function simulate_nonlinear(
    f::Function,
    Q,
    h::Function,
    R,
    μ₀,
    Σ₀,
    N::Int;
    rng = Random.GLOBAL_RNG,
)
    x = rand(rng, MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(rng, MvNormal(f(states[end]), Q(x))))
        push!(observations, rand(rng, MvNormal(h(states[end]), R(x))))
    end
    return states, observations
end

stack(x) = copy(reduce(hcat, x)')

function filtering_setup()
    d, D = 1, 2
    μ₀ = [-1.0, 1.0]
    Σ₀ = [0.01 0.0
        0.0 0.01]
    a, b, c = 0.2, 0.2, 3.0

    function f(x)
        x1, x2 = x
        return [
            x1 + 0.1 * (c * (x1 - x1^3 / 3 + x2)),
            x2 + 0.1 * (-(1 / c) * (x1 - a - b * x2)),
        ]
    end
    function h(x)
        return x[1:1]
    end

    A(x) = ForwardDiff.jacobian(f, x)
    Q(x) = Matrix{Float64}(0.001 * I(D))
    H(x) = Matrix{Float64}(I(D))[1:1, :]
    R(x) = Matrix{Float64}(I(d))
    u(x) = f(x) - A(x) * x
    v(x) = zeros(d)

    N = 200
    ground_truth, observations = simulate_nonlinear(f, Q, h, R, μ₀, Σ₀, N)

    return μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations
end

@info "Executing tests for standard Kalman filters"
include("algorithms/kalman.jl")
@info "Executing tests for square-root Kalman filters"
include("algorithms/sqrt_kalman.jl")
@info "Executing tests for ensemble Kalman filters"
include("algorithms/enkf.jl")
