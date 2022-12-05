using Test
using LinearAlgebra
using StatsBase
using Statistics
using Random
using Distributions
using ForwardDiff

using PhDSE


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
    d, D = 2, 2
    μ₀ = [-1.0, 1.0]
    Σ₀ = [0.01 0.0
          0.0 0.01]
    a, b, c = 0.2, 0.2, 3.0

    function f(x)
        x1, x2 = x
        return [
            x1 + 0.1 * (c * (x1 - x1^3 / 3 + x2)),
            x2 + 0.1 * (-(1 / c) * (x1 - a - b * x2))
        ]
    end
    function h(x)
        return copy(x)
    end

    A(x) = ForwardDiff.jacobian(f, x)
    Q(x) = Matrix{Float64}(0.001 * I(D))
    H(x) = Matrix{Float64}(I(d))
    R(x) = Matrix{Float64}(I(d))
    u(x) = f(x) - A(x) * x
    v(x) = zeros(d)

    N = 200
    ground_truth, observations = simulate_nonlinear(f, Q, h, R, μ₀, Σ₀, N)

    return μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations
end

@testset "Kalman filter" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache(initial_mean=μ₀, initial_covariance=Σ₀)

    iip_m = copy(μ₀)
    oop_m = copy(μ₀)
    iip_C = copy(Σ₀)
    oop_C = copy(Σ₀)
    for y in observations
        iip_m, iip_C = kf_predict!(cache, A(iip_m), Q(iip_m), u(iip_m))
        oop_m, oop_C = kf_predict(oop_m, oop_C, A(oop_m), Q(oop_m), u(oop_m))
        # @show iip_m oop_m
        @assert iip_m ≈ oop_m
        @assert iip_C ≈ oop_C

        iip_m, iip_C = kf_correct!(cache, H(iip_m), R(iip_m), y, v(iip_m))
        oop_m, oop_C = kf_joseph_correct(oop_m, oop_C, H(oop_m), R(oop_m), y, v(oop_m))
        # @show iip_m oop_m
        @assert iip_m ≈ oop_m
        @assert iip_C ≈ oop_C
    end

end

# @testset "Square-root Kalman filter" begin
#     Random.seed!(1234)

#     Φ = randn(5, 5)
#     Q = Matrix(I(5) * 0.1)
#     u = randn(5)

#     H = randn(3, 5)
#     r = randn(3, 3)
#     R = (r' * r) .+ 1e-2 .* Matrix(I(3))
#     v = randn(3)

#     y = [1.0, 2.0, 3.0]

#     m0, C0 = randn(5), Matrix(I(5) * 0.07)

#     QU = cholesky(Q).U
#     RU = cholesky(R).U

#     kf_cache = KFCache(5, 3)
#     write_moments!(kf_cache; μ = m0, Σ = C0)

#     sqkf_cache = SqrtKFCache(5, 3)
#     write_moments!(sqkf_cache; μ = m0, Σ = C0)

#     # PREDICT
#     kf_predict!(kf_cache, Φ, Q, u)
#     sqrt_kf_predict!(sqkf_cache, Φ, QU, u)

#     @test kf_cache.μ⁻ ≈ sqkf_cache.μ⁻
#     @test kf_cache.Σ⁻ ≈ sqkf_cache.Σ⁻' * sqkf_cache.Σ⁻

#     # UPDATE
#     kf_correct!(kf_cache, H, R, y, v)
#     sqrt_kf_correct!(sqkf_cache, H, RU, y, v)

#     @test kf_cache.Σ ≈ sqkf_cache.Σ' * sqkf_cache.Σ
#     @test kf_cache.μ ≈ sqkf_cache.μ
# end

# @testset "Ensemble Kalman filter (EnKF)" begin
#     Random.seed!(1234)

#     # dynamics
#     A, Q = ones(1, 1), 0.01 .* ones(1, 1)
#     u = 0.001 .* ones(1)

#     # observations
#     H, R = ones(1, 1), 0.025 .* I(1)
#     v = 0.001 .* ones(1)
#     d, D = size(H)

#     gt = [sin(t / 10) for t in 1:100]
#     artificial_data = [s .+ 0.05 .* randn(1) for s in copy(gt)[2:end]]

#     N = 25

#     # initial conditions
#     μ₀, Σ₀ = gt[1] .* ones(1), 0.05 .* ones(1, 1)

#     fcache = EnKFCache(
#         D,
#         d,
#         ensemble_size = N,
#         process_noise_dist = MvNormal(zeros(D), Q),
#         observation_noise_dist = MvNormal(zeros(d), R),
#     )
#     write_moments!(fcache; μ = μ₀, Σ = Σ₀)

#     sol = [(copy(μ₀), copy(Σ₀))]
#     for y in artificial_data
#         enkf_predict!(fcache, A, u)
#         enkf_correct!(fcache, H, inv(R), y, v)
#         push!(
#             sol,
#             (mean(eachcol(fcache.ensemble)), cov(fcache.ensemble, dims = 2)),
#         )
#     end

#     rmse = rmsd([m[1] for (m, S) in sol], gt)
#     @test rmse < 0.1
# end
