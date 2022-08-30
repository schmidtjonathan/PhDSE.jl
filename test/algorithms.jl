using Test
using LinearAlgebra
using StatsBase
using Random

using PhDSE


@testset "Kalman filter" begin
    Random.seed!(1234)

    # dynamics
    A, Q = ones(1, 1), 0.01 .* ones(1, 1)

    # observations
    H, R = ones(1, 1), 0.025 .* I(1)
    d, D = size(H)

    gt = [sin(t / 10) for t in 1:100]
    artificial_data = [s .+ 0.05 .* randn(1) for s in copy(gt)[2:end]]

    # initial conditions
    μ₀, Σ₀ = gt[1] .* ones(1), 0.05 .* ones(1, 1)

    sol = [(copy(μ₀), copy(Σ₀))]
    fcache = KFCache(D, d)
    fcache.μ .= μ₀
    fcache.Σ .= Σ₀
    for y in artificial_data
        kf_predict!(fcache, A, Q)
        kf_correct!(fcache, H, R, y)
        push!(sol, (copy(fcache.μ), copy(fcache.Σ)))
    end

    rmse = rmsd([m[1] for (m, S) in sol], gt)
    @test rmse < 0.1
end