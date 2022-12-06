using Test
using LinearAlgebra
using StatsBase
using Statistics
using Random
using Distributions
using ForwardDiff

using PhDSE

const PLOT_RESULTS = true

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
            x2 + 0.1 * (-(1 / c) * (x1 - a - b * x2)),
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

@testset "Kalman filter IIP vs. OOP" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache(initial_mean = μ₀, initial_covariance = Σ₀)

    iip_m = copy(μ₀)
    oop_m = copy(μ₀)
    iip_C = copy(Σ₀)
    oop_C = copy(Σ₀)
    iip_traj = [(copy(μ₀), copy(Σ₀))]
    oop_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        iip_m, iip_C = kf_predict!(cache, A(iip_m), Q(iip_m), u(iip_m))
        oop_m, oop_C = kf_predict(oop_m, oop_C, A(oop_m), Q(oop_m), u(oop_m))
        # @test iip_m ≈ oop_m
        # @test iip_C ≈ oop_C

        iip_m, iip_C = kf_correct!(cache, H(iip_m), R(iip_m), y, v(iip_m))
        oop_m, oop_C = kf_correct(oop_m, oop_C, H(oop_m), R(oop_m), y, v(oop_m))
        # @test iip_m ≈ oop_m
        # @test iip_C ≈ oop_C
        push!(iip_traj, (copy(iip_m), copy(iip_C)))
        push!(oop_traj, (copy(oop_m), copy(oop_C)))
    end

    @test all([m1 ≈ m2 for ((m1, C1), (m2, C2)) in zip(iip_traj, oop_traj)])
    @test all([C1 ≈ C2 for ((m1, C1), (m2, C2)) in zip(iip_traj, oop_traj)])

    if PLOT_RESULTS
        iip_means = [m for (m, C) in iip_traj]
        oop_means = [m for (m, C) in oop_traj]
        iip_stds = [2sqrt.(diag(C)) for (m, C) in iip_traj]
        oop_stds = [2sqrt.(diag(C)) for (m, C) in oop_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(iip_means),
            [m[1] for m in iip_means],
            ribbon = [s[1] for s in iip_stds],
            label = "iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(iip_means),
            [m[2] for m in iip_means],
            ribbon = [s[2] for s in iip_stds],
            label = "iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(oop_means),
            [m[1] for m in oop_means],
            ribbon = [s[1] for s in oop_stds],
            label = "oop",
            color = 4,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(oop_means),
            [m[2] for m in oop_means],
            ribbon = [s[2] for s in oop_stds],
            label = "oop",
            color = 4,
            lw = 3,
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "kf_test_output.png"))
    end
end

@testset "Kalman update vs. Joseph update" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()

    standard_m = copy(μ₀)
    joseph_m = copy(μ₀)
    standard_C = copy(Σ₀)
    joseph_C = copy(Σ₀)
    standard_traj = [(copy(μ₀), copy(Σ₀))]
    joseph_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        standard_m, standard_C =
            kf_predict(standard_m, standard_C, A(standard_m), Q(standard_m), u(standard_m))
        joseph_m, joseph_C =
            kf_predict(joseph_m, joseph_C, A(joseph_m), Q(joseph_m), u(joseph_m))

        standard_m, standard_C = kf_correct(
            standard_m,
            standard_C,
            H(standard_m),
            R(standard_m),
            y,
            v(standard_m),
        )
        joseph_m, joseph_C =
            kf_joseph_correct(joseph_m, joseph_C, H(joseph_m), R(joseph_m), y, v(joseph_m))

        push!(standard_traj, (copy(standard_m), copy(standard_C)))
        push!(joseph_traj, (copy(joseph_m), copy(joseph_C)))
    end

    @test all([m1 ≈ m2 for ((m1, C1), (m2, C2)) in zip(standard_traj, joseph_traj)])
    @test all([C1 ≈ C2 for ((m1, C1), (m2, C2)) in zip(standard_traj, joseph_traj)])

    if PLOT_RESULTS
        standard_means = [m for (m, C) in standard_traj]
        joseph_means = [m for (m, C) in joseph_traj]
        standard_stds = [2sqrt.(diag(C)) for (m, C) in standard_traj]
        joseph_stds = [2sqrt.(diag(C)) for (m, C) in joseph_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(standard_means),
            [m[1] for m in standard_means],
            ribbon = [s[1] for s in standard_stds],
            label = "standard",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(standard_means),
            [m[2] for m in standard_means],
            ribbon = [s[2] for s in standard_stds],
            label = "standard",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(joseph_means),
            [m[1] for m in joseph_means],
            ribbon = [s[1] for s in joseph_stds],
            label = "joseph",
            color = 4,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(joseph_means),
            [m[2] for m in joseph_means],
            ribbon = [s[2] for s in joseph_stds],
            label = "joseph",
            color = 4,
            lw = 3,
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "kf_joseph_test_output.png"))
    end
end

@testset "Kalman filter (OOP) vs. Sqrt-KF (IIP + OOP)" begin
    Random.seed!(1234)
    upper_sqrt_to_mat(MU::UpperTriangular) = MU' * MU

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache(initial_mean = μ₀, initial_covariance = cholesky(Σ₀).U)

    kf_m = copy(μ₀)
    iip_sqrt_kf_m = copy(μ₀)
    oop_sqrt_kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    iip_sqrt_kf_C = cholesky(Σ₀).U
    oop_sqrt_kf_C = cholesky(Σ₀).U
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    iip_traj = [(copy(μ₀), copy(Σ₀))]
    oop_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        kf_m, kf_C = kf_predict(kf_m, kf_C, A(kf_m), Q(kf_m), u(kf_m))
        iip_sqrt_kf_m, iip_sqrt_kf_C = sqrt_kf_predict!(
            cache,
            A(iip_sqrt_kf_m),
            cholesky(Q(iip_sqrt_kf_m)).U,
            u(iip_sqrt_kf_m),
        )
        oop_sqrt_kf_m, oop_sqrt_kf_C = sqrt_kf_predict(
            oop_sqrt_kf_m,
            oop_sqrt_kf_C,
            A(oop_sqrt_kf_m),
            cholesky(Q(oop_sqrt_kf_m)).U,
            u(oop_sqrt_kf_m),
        )
        # @test kf_m ≈ iip_sqrt_kf_m ≈ oop_sqrt_kf_m
        # @test kf_C ≈ upper_sqrt_to_mat(iip_sqrt_kf_C) ≈ upper_sqrt_to_mat(oop_sqrt_kf_C)

        kf_m, kf_C = kf_correct(kf_m, kf_C, H(kf_m), R(kf_m), y, v(kf_m))
        iip_sqrt_kf_m, iip_sqrt_kf_C = sqrt_kf_correct!(
            cache,
            H(iip_sqrt_kf_m),
            cholesky(R(iip_sqrt_kf_m)).U,
            y,
            v(iip_sqrt_kf_m),
        )
        oop_sqrt_kf_m, oop_sqrt_kf_C = sqrt_kf_correct(
            oop_sqrt_kf_m,
            oop_sqrt_kf_C,
            H(oop_sqrt_kf_m),
            cholesky(R(oop_sqrt_kf_m)).U,
            y,
            v(oop_sqrt_kf_m),
        )
        # @test kf_m ≈ iip_sqrt_kf_m ≈ oop_sqrt_kf_m
        # @test kf_C ≈ upper_sqrt_to_mat(iip_sqrt_kf_C) ≈ upper_sqrt_to_mat(oop_sqrt_kf_C)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
        push!(iip_traj, (copy(iip_sqrt_kf_m), upper_sqrt_to_mat(iip_sqrt_kf_C)))
        push!(oop_traj, (copy(oop_sqrt_kf_m), upper_sqrt_to_mat(oop_sqrt_kf_C)))
    end

    @test all([
        m1 ≈ m2 ≈ m3 for ((m1, C1), (m2, C2), (m3, C3)) in zip(kf_traj, iip_traj, oop_traj)
    ])
    @test all([
        C1 ≈ C2 ≈ C3 for ((m1, C1), (m2, C2), (m3, C3)) in zip(kf_traj, iip_traj, oop_traj)
    ])

    if PLOT_RESULTS
        kf_means = [m for (m, C) in kf_traj]
        iip_means = [m for (m, C) in iip_traj]
        oop_means = [m for (m, C) in oop_traj]
        kf_stds = [2sqrt.(diag(C)) for (m, C) in kf_traj]
        iip_stds = [2diag(C) for (m, C) in iip_traj]
        oop_stds = [2diag(C) for (m, C) in oop_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(iip_means),
            [m[1] for m in iip_means],
            ribbon = [s[1] for s in iip_stds],
            label = "iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(iip_means),
            [m[2] for m in iip_means],
            ribbon = [s[2] for s in iip_stds],
            label = "iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(oop_means),
            [m[1] for m in oop_means],
            ribbon = [s[1] for s in oop_stds],
            label = "oop",
            color = 4,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(oop_means),
            [m[2] for m in oop_means],
            ribbon = [s[2] for s in oop_stds],
            label = "oop",
            color = 4,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(kf_means),
            [m[1] for m in kf_means],
            ribbon = [s[1] for s in kf_stds],
            label = "kf",
            color = 5,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(kf_means),
            [m[2] for m in kf_means],
            ribbon = [s[2] for s in kf_stds],
            label = "kf",
            color = 5,
            lw = 3,
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "sqrt_kf_test_output.png"))
    end
end

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
