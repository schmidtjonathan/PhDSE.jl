using Test
using LinearAlgebra
using StatsBase
using Random
using PSDMatrices
using Distributions

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

@testset "Square-root Kalman filter" begin
    Random.seed!(1234)

    Φ = randn(5, 5)
    Q = Matrix(I(5) * 0.1)

    H = randn(3, 5)
    r = randn(3, 3)
    R = (r' * r) .+ 1e-2 .* Matrix(I(3))

    y = [1.0, 2.0, 3.0]

    m0, C0 = randn(5), Matrix(I(5) * 0.07)

    QU = PSDMatrix(cholesky(Q).U)
    RU = PSDMatrix(cholesky(R).U)
    C0U = PSDMatrix(cholesky(C0).U)

    kf_cache = KFCache(5, 3)
    kf_cache.μ .= m0
    kf_cache.Σ .= C0

    sqkf_cache = SqrtKFCache(5, 3)
    sqkf_cache.μ .= m0
    copy!(sqkf_cache.Σ.R, C0U.R)

    # PREDICT
    kf_predict!(kf_cache, Φ, Q)
    sqrt_kf_predict!(sqkf_cache, Φ, QU)

    @test kf_cache.μ⁻ ≈ sqkf_cache.μ⁻
    @test kf_cache.Σ⁻ ≈ Matrix(sqkf_cache.Σ⁻)

    # UPDATE
    kf_correct!(kf_cache, H, R, y)
    sqrt_kf_correct!(sqkf_cache, H, RU, y)

    @test kf_cache.Σ ≈ Matrix(sqkf_cache.Σ)
    @test kf_cache.μ ≈ sqkf_cache.μ
end

@testset "Ensemble Kalman filter (EnKF)" begin
    Random.seed!(1234)

    # dynamics
    A, Q = ones(1, 1), 0.01 .* ones(1, 1)

    # observations
    H, R = ones(1, 1), 0.025 .* I(1)
    d, D = size(H)

    gt = [sin(t / 10) for t in 1:100]
    artificial_data = [s .+ 0.05 .* randn(1) for s in copy(gt)[2:end]]

    N = 25

    # initial conditions
    μ₀, Σ₀ = gt[1] .* ones(1), 0.05 .* ones(1, 1)
    init_ensemble = rand(MvNormal(μ₀, Σ₀), N)

    sol = [(copy(μ₀), copy(Σ₀))]
    fcache = EnKFCache(
        D,
        d,
        ensemble_size = N,
        process_noise_dist = MvNormal(zeros(D), Q),
        observation_noise_dist = MvNormal(zeros(d), R),
    )
    copy!(fcache.ensemble, init_ensemble)
    for y in artificial_data
        enkf_predict!(fcache, A)
        enkf_correct!(fcache, H, inv(R), y)
        push!(
            sol,
            (PhDSE.ensemble_mean(fcache.ensemble), PhDSE.ensemble_cov(fcache.ensemble)),
        )
    end

    rmse = rmsd([m[1] for (m, S) in sol], gt)
    @test rmse < 0.1
end
