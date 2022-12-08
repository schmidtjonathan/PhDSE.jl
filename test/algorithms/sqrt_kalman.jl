@testset "Kalman filter (OOP) vs. SqrtKF (IIP) vs. SqrtKF (OOP)" begin
    Random.seed!(1234)
    upper_sqrt_to_mat(MU::UpperTriangular) = MU' * MU

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache(initial_mean = μ₀, initial_covariance = cholesky(Σ₀).U)
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "mean"))
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "predicted_mean"))
    @test haskey(cache.entries, (typeof(Σ₀), size(Σ₀), "covariance"))
    @test haskey(cache.entries, (typeof(Σ₀), size(Σ₀), "predicted_covariance"))

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
            label = "sqrt iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(iip_means),
            [m[2] for m in iip_means],
            ribbon = [s[2] for s in iip_stds],
            label = "sqrt iip",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(oop_means),
            [m[1] for m in oop_means],
            ribbon = [s[1] for s in oop_stds],
            label = "sqrt oop",
            color = 4,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(oop_means),
            [m[2] for m in oop_means],
            ribbon = [s[2] for s in oop_stds],
            label = "sqrt oop",
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
        savefig(test_plot, joinpath(mkpath("./out/"), "kf_oop-vs-sqrtkf_iip-vs-sqrtkf_oop.png"))
    end
    # for (k, v) in pairs(cache.entries)
    #     println("$k -> $(typeof(v)) of size $(size(v))")
    # end
end
