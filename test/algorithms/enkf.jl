const ENSEMBLE_SIZE = 1000

@testset "Kalman filter (OOP) vs. standard EnKF (OOP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist(x) = MvNormal(zero(x), Q(x))
    measurement_noise_dist(x) = MvNormal(zero(x), R(x))
    ensemble = rand(init_dist, ENSEMBLE_SIZE)

    kf_m = copy(μ₀)
    enkf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    enkf_C = cholesky(Σ₀).U
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    enkf_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        kf_m, kf_C = kf_predict(kf_m, kf_C, A(kf_m), Q(kf_m), u(kf_m))
        ensemble = enkf_predict(
            ensemble,
            A(enkf_m),
            process_noise_dist(enkf_m),
            u(enkf_m),
        )
        enkf_m, enkf_C = ensemble_mean_cov(ensemble)

        kf_m, kf_C = kf_correct(kf_m, kf_C, H(kf_m), R(kf_m), y, v(kf_m))
        ensemble = enkf_correct(
            ensemble,
            H(enkf_m),
            measurement_noise_dist(enkf_m),
            y,
            v(enkf_m),
        )
        enkf_m, enkf_C = ensemble_mean_cov(ensemble)

        push!(kf_traj, (copy(kf_m), copy(kf_C)))
        push!(enkf_traj, (copy(enkf_m), copy(enkf_C)))
    end

    # for ((m1, C1), (m2, C2)) in zip(kf_traj, enkf_traj)
    #     println("$m1 vs. $m2")
    # end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(kf_traj, enkf_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(kf_traj, enkf_traj)
    ])

    if PLOT_RESULTS
        kf_means = [m for (m, C) in kf_traj]
        enkf_means = [m for (m, C) in enkf_traj]
        kf_stds = [2sqrt.(diag(C)) for (m, C) in kf_traj]
        enkf_stds = [2diag(C) for (m, C) in enkf_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(enkf_means),
            [m[1] for m in enkf_means],
            ribbon = [s[1] for s in enkf_stds],
            label = "enkf",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(enkf_means),
            [m[2] for m in enkf_means],
            ribbon = [s[2] for s in enkf_stds],
            label = "enkf",
            color = 3,
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
        savefig(test_plot, joinpath(mkpath("./out/"), "enkf_standard_oop_test_output.png"))
    end
end
