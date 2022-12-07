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
    enkf_C = copy(Σ₀)
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
        enkf_stds = [2sqrt.(diag(C)) for (m, C) in enkf_traj]
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
            ls=:dot
        )
        plot!(
            test_plot2,
            1:length(kf_means),
            [m[2] for m in kf_means],
            ribbon = [s[2] for s in kf_stds],
            label = "kf",
            color = 5,
            lw = 3,
            ls=:dot
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "enkf_standard_oop_test_output.png"))
    end
end


@testset "Standard EnKF (OOP) vs. observation-matrix-free EnKF (OOP) with N > d" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist(x) = MvNormal(zero(x), Q(x))
    measurement_noise_dist(x) = MvNormal(zero(x), R(x))
    standard_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)
    omf_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)

    standard_m = copy(μ₀)
    omf_m = copy(μ₀)
    standard_C = copy(Σ₀)
    omf_C = copy(Σ₀)
    standard_traj = [(copy(μ₀), copy(Σ₀))]
    omf_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        standard_ensemble = enkf_predict(
            standard_ensemble,
            A(standard_m),
            process_noise_dist(standard_m),
            u(standard_m),
        )
        omf_ensemble = enkf_predict(
            omf_ensemble,
            A(omf_m),
            process_noise_dist(omf_m),
            u(omf_m),
        )
        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        standard_ensemble = enkf_correct(
            standard_ensemble,
            H(standard_m),
            measurement_noise_dist(y),
            y,
            v(standard_m),
        )

        # _A = centered_ensemble(omf_ensemble)
        HX =  H(omf_m) * omf_ensemble .+ v(omf_m)
        HA =  centered_ensemble(HX)
        omf_ensemble = enkf_matrixfree_correct(
            omf_ensemble,
            HX,
            HA,
            measurement_noise_dist(y),
            y;
            # A = _A,
            R_inverse=missing
        )
        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        push!(standard_traj, (copy(standard_m), copy(standard_C)))
        push!(omf_traj, (copy(omf_m), copy(omf_C)))
    end

    # for ((m1, C1), (m2, C2)) in zip(standard_traj, omf_traj)
    #     println("$m1 vs. $m2")
    # end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(standard_traj, omf_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(standard_traj, omf_traj)
    ])

    if PLOT_RESULTS
        standard_means = [m for (m, C) in standard_traj]
        omf_means = [m for (m, C) in omf_traj]
        standard_stds = [2sqrt.(diag(C)) for (m, C) in standard_traj]
        omf_stds = [2sqrt.(diag(C)) for (m, C) in omf_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(omf_means),
            [m[1] for m in omf_means],
            ribbon = [s[1] for s in omf_stds],
            label = "OMF",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(omf_means),
            [m[2] for m in omf_means],
            ribbon = [s[2] for s in omf_stds],
            label = "OMF",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(standard_means),
            [m[1] for m in standard_means],
            ribbon = [s[1] for s in standard_stds],
            label = "standard",
            color = 5,
            lw = 3,
            ls=:dot
        )
        plot!(
            test_plot2,
            1:length(standard_means),
            [m[2] for m in standard_means],
            ribbon = [s[2] for s in standard_stds],
            label = "standard",
            color = 5,
            lw = 3,
            ls=:dot
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "omf_vs_standard_oop_test_output.png"))
    end
end


@testset "OMF EnKF with P (OOP) vs. P⁻¹ (OOP) with N < d" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist(x) = MvNormal(zero(x), Q(x))
    measurement_noise_dist(x) = MvNormal(zero(x), R(x))
    mil_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)
    omf_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)

    mil_m = copy(μ₀)
    omf_m = copy(μ₀)
    mil_C = copy(Σ₀)
    omf_C = copy(Σ₀)
    mil_traj = [(copy(μ₀), copy(Σ₀))]
    omf_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        mil_ensemble = enkf_predict(
            mil_ensemble,
            A(mil_m),
            process_noise_dist(mil_m),
            u(mil_m),
        )
        omf_ensemble = enkf_predict(
            omf_ensemble,
            A(omf_m),
            process_noise_dist(omf_m),
            u(omf_m),
        )
        mil_m, mil_C = ensemble_mean_cov(mil_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        mil_HX =  H(mil_m) * mil_ensemble .+ v(mil_m)
        mil_HA =  centered_ensemble(mil_HX)
        mil_ensemble = enkf_matrixfree_correct(
            mil_ensemble,
            mil_HX,
            mil_HA,
            measurement_noise_dist(y),
            y;
            # A = _A,
            R_inverse=inv(R(y))
        )

        omf_HX =  H(omf_m) * omf_ensemble .+ v(omf_m)
        omf_HA =  centered_ensemble(omf_HX)
        omf_ensemble = enkf_matrixfree_correct(
            omf_ensemble,
            omf_HX,
            omf_HA,
            measurement_noise_dist(y),
            y;
            # A = _A,
            R_inverse=missing
        )

        mil_m, mil_C = ensemble_mean_cov(mil_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        push!(mil_traj, (copy(mil_m), copy(mil_C)))
        push!(omf_traj, (copy(omf_m), copy(omf_C)))
    end

    # for ((m1, C1), (m2, C2)) in zip(mil_traj, omf_traj)
    #     println("$m1 vs. $m2")
    # end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(mil_traj, omf_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(mil_traj, omf_traj)
    ])

    if PLOT_RESULTS
        mil_means = [m for (m, C) in mil_traj]
        omf_means = [m for (m, C) in omf_traj]
        mil_stds = [2sqrt.(diag(C)) for (m, C) in mil_traj]
        omf_stds = [2sqrt.(diag(C)) for (m, C) in omf_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        test_plot2 =
            scatter(1:length(observations), [o[2] for o in observations], color = 2)
        plot!(
            test_plot1,
            1:length(omf_means),
            [m[1] for m in omf_means],
            ribbon = [s[1] for s in omf_stds],
            label = "OMF",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(omf_means),
            [m[2] for m in omf_means],
            ribbon = [s[2] for s in omf_stds],
            label = "OMF",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(mil_means),
            [m[1] for m in mil_means],
            ribbon = [s[1] for s in mil_stds],
            label = "MIL",
            color = 5,
            lw = 3,
            ls=:dot
        )
        plot!(
            test_plot2,
            1:length(mil_means),
            [m[2] for m in mil_means],
            ribbon = [s[2] for s in mil_stds],
            label = "MIL",
            color = 5,
            lw = 3,
            ls=:dot
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(test_plot, joinpath(mkpath("./out/"), "omf_vs_mil_oop_test_output.png"))
    end
end
