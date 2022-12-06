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