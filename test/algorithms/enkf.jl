const ENSEMBLE_SIZE = 2000

@testset "Kalman filter (OOP) vs. standard EnKF (OOP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(ground_truth[1]), Q)
    measurement_noise_dist = MvNormal(zero(observations[1]), R)
    # Choose larger ensemble size when comparing to exact KF computation
    ensemble = rand(Xoshiro(123), init_dist, ENSEMBLE_SIZE * 15)

    kf_m = copy(μ₀)
    enkf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    enkf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    enkf_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        kf_m, kf_C = kf_predict(kf_m, kf_C, A, Q, u)
        ensemble = enkf_predict(
            ensemble,
            A,
            process_noise_dist,
            u,
        )
        enkf_m, enkf_C = ensemble_mean_cov(ensemble)

        kf_m, kf_C = kf_correct(kf_m, kf_C, H, R, y, v)
        ensemble = enkf_correct(
            ensemble,
            H,
            measurement_noise_dist,
            y,
            v,
        )
        enkf_m, enkf_C = ensemble_mean_cov(ensemble)

        push!(kf_traj, (copy(kf_m), copy(kf_C)))
        push!(enkf_traj, (copy(enkf_m), copy(enkf_C)))
    end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(kf_traj, enkf_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(kf_traj, enkf_traj)
    ])

    if PLOT_RESULTS
        kf_means = stack([m for (m, C) in kf_traj])
        enkf_means = stack([m for (m, C) in enkf_traj])
        kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj])
        enkf_stds = stack([2sqrt.(diag(C)) for (m, C) in enkf_traj])

        out_dir = mkpath("./out/KF_oop-vs-standardEnKF_oop")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=kf_means, estim_stds=kf_stds),
            joinpath(out_dir, "kf.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=enkf_means, estim_stds=enkf_stds),
            joinpath(out_dir, "enkf.png")
        )
    end
end

@testset "Standard EnKF (OOP) vs. O(d^3) OMF EnKF (OOP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(ground_truth[1]), Q)
    measurement_noise_dist = MvNormal(zero(observations[1]), R)
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
            A,
            process_noise_dist,
            u,
        )
        omf_ensemble = enkf_predict(
            omf_ensemble,
            A,
            process_noise_dist,
            u,
        )
        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        standard_ensemble = enkf_correct(
            standard_ensemble,
            H,
            measurement_noise_dist,
            y,
            v,
        )

        # _A = centered_ensemble(omf_ensemble)
        HX = H * omf_ensemble .+ v
        HA = centered_ensemble(HX)
        omf_ensemble = enkf_matrixfree_correct(
            omf_ensemble,
            HX,
            HA,
            measurement_noise_dist,
            y;
            # A = _A,
            R_inverse = missing,
        )
        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        push!(standard_traj, (copy(standard_m), copy(standard_C)))
        push!(omf_traj, (copy(omf_m), copy(omf_C)))
    end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(standard_traj, omf_traj)
    ])
    # @test all([
    #     isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
    #     ((m1, C1), (m2, C2)) in zip(standard_traj, omf_traj)
    # ])


    if PLOT_RESULTS
        standard_means = stack([m for (m, C) in standard_traj])
        omf_means = stack([m for (m, C) in omf_traj])
        standard_stds = stack([2sqrt.(diag(C)) for (m, C) in standard_traj])
        omf_stds = stack([2sqrt.(diag(C)) for (m, C) in omf_traj])

        out_dir = mkpath("./out/standardEnKF_oop-vs-d3OMFEnKF_oop")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=standard_means, estim_stds=standard_stds),
            joinpath(out_dir, "kf.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=omf_means, estim_stds=omf_stds),
            joinpath(out_dir, "enkf.png")
        )
    end
end

@testset "O(d^3) OMF EnKF (OOP) vs. O(N^3) OMF EnKF (OOP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(ground_truth[1]), Q)
    measurement_noise_dist = MvNormal(zero(observations[1]), R)
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
            A,
            process_noise_dist,
            u,
        )
        omf_ensemble = enkf_predict(
            omf_ensemble,
            A,
            process_noise_dist,
            u,
        )
        mil_m, mil_C = ensemble_mean_cov(mil_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        mil_HX = H * mil_ensemble .+ v
        mil_HA = centered_ensemble(mil_HX)
        mil_ensemble = enkf_matrixfree_correct(
            mil_ensemble,
            mil_HX,
            mil_HA,
            measurement_noise_dist,
            y;
            # A = _A,
            R_inverse = inv(R),
        )

        omf_HX = H * omf_ensemble .+ v
        omf_HA = centered_ensemble(omf_HX)
        omf_ensemble = enkf_matrixfree_correct(
            omf_ensemble,
            omf_HX,
            omf_HA,
            measurement_noise_dist,
            y;
            # A = _A,
            R_inverse = missing,
        )

        mil_m, mil_C = ensemble_mean_cov(mil_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)

        push!(mil_traj, (copy(mil_m), copy(mil_C)))
        push!(omf_traj, (copy(omf_m), copy(omf_C)))
    end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(mil_traj, omf_traj)
    ])
    # @test all([
    #     isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
    #     ((m1, C1), (m2, C2)) in zip(mil_traj, omf_traj)
    # ])

    if PLOT_RESULTS
        mil_means = stack([m for (m, C) in mil_traj])
        omf_means = stack([m for (m, C) in omf_traj])
        mil_stds = stack([2sqrt.(diag(C)) for (m, C) in mil_traj])
        omf_stds = stack([2sqrt.(diag(C)) for (m, C) in omf_traj])

        out_dir = mkpath("./out/d3OMFEnKF_oop-vs-N3OMFEnKF_oop")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=mil_means, estim_stds=mil_stds),
            joinpath(out_dir, "ON3.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=omf_means, estim_stds=omf_stds),
            joinpath(out_dir, "OD3.png")
        )
    end
end

@testset "Standard EnKF (OOP) vs. Standard EnKF (IIP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist(x) = MvNormal(zero(x), Q(x))
    measurement_noise_dist(x) = MvNormal(zero(x), R(x))
    oop_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)
    iip_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)

    enkf_cache = FilteringCache()
    init_cache_ensemble!(enkf_cache, iip_ensemble)
    @test haskey(
        enkf_cache.entries,
        (typeof(size(iip_ensemble, 2)), size(size(iip_ensemble, 2)), "N"),
    )
    @test haskey(enkf_cache.entries, (typeof(iip_ensemble), size(iip_ensemble), "ensemble"))
    @test haskey(
        enkf_cache.entries,
        (typeof(iip_ensemble), size(iip_ensemble), "forecast_ensemble"),
    )

    @assert iip_ensemble ==
            enkf_cache.entries[(typeof(iip_ensemble), size(iip_ensemble), "ensemble")]

    oop_m = copy(μ₀)
    iip_m = copy(μ₀)
    oop_C = copy(Σ₀)
    iip_C = copy(Σ₀)
    oop_traj = [(copy(μ₀), copy(Σ₀))]
    iip_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        oop_ensemble = enkf_predict(
            oop_ensemble,
            A(oop_m),
            process_noise_dist(oop_m),
            u(oop_m),
        )
        iip_ensemble = enkf_predict!(
            enkf_cache,
            A(iip_m),
            process_noise_dist(iip_m),
            u(iip_m),
        )
        @assert iip_ensemble === enkf_cache.entries[(
            typeof(iip_ensemble),
            size(iip_ensemble),
            "forecast_ensemble",
        )]
        # @show oop_m
        # @show iip_m
        oop_m, oop_C = ensemble_mean_cov(oop_ensemble)
        iip_m, iip_C = ensemble_mean_cov(copy(iip_ensemble))

        oop_ensemble = enkf_correct(
            oop_ensemble,
            H(oop_m),
            measurement_noise_dist(y),
            y,
            v(oop_m),
        )

        iip_ensemble = enkf_correct!(
            enkf_cache,
            H(iip_m),
            measurement_noise_dist(y),
            y,
            v(iip_m),
        )
        oop_m, oop_C = ensemble_mean_cov(oop_ensemble)
        iip_m, iip_C = ensemble_mean_cov(iip_ensemble)

        push!(oop_traj, (copy(oop_m), copy(oop_C)))
        push!(iip_traj, (copy(iip_m), copy(iip_C)))
    end

    # for ((m1, C1), (m2, C2)) in zip(oop_traj, iip_traj)
    #     println("$m1 vs. $m2")
    # end

    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(oop_traj, iip_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(oop_traj, iip_traj)
    ])

    if PLOT_RESULTS
        oop_means = [m for (m, C) in oop_traj]
        iip_means = [m for (m, C) in iip_traj]
        oop_stds = [2sqrt.(diag(C)) for (m, C) in oop_traj]
        iip_stds = [2sqrt.(diag(C)) for (m, C) in iip_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        plot!(
            test_plot1,
            1:length(ground_truth),
            [gt[1] for gt in ground_truth],
            label = "gt",
            color = :black,
            lw = 5,
            alpha = 0.4,
        )
        test_plot2 = plot(
            1:length(ground_truth),
            [gt[2] for gt in ground_truth],
            label = "gt",
            color = :black,
            lw = 5,
            alpha = 0.4,
        )
        plot!(
            test_plot1,
            1:length(iip_means),
            [m[1] for m in iip_means],
            ribbon = [s[1] for s in iip_stds],
            label = "IIP",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(iip_means),
            [m[2] for m in iip_means],
            ribbon = [s[2] for s in iip_stds],
            label = "IIP",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(oop_means),
            [m[1] for m in oop_means],
            ribbon = [s[1] for s in oop_stds],
            label = "OOP",
            color = 5,
            lw = 3,
            ls = :dot,
        )
        plot!(
            test_plot2,
            1:length(oop_means),
            [m[2] for m in oop_means],
            ribbon = [s[2] for s in oop_stds],
            label = "OOP",
            color = 5,
            lw = 3,
            ls = :dot,
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(
            test_plot,
            joinpath(mkpath("./out/"), "standardEnKF_oop-vs-standardEnKF_iip.png"),
        )
    end
end

@testset "Standard EnKF (IIP) vs. O(d^3) OMF EnKF (IIP) vs. O(N^3) OMF EnKF (IIP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist(x) = MvNormal(zero(x), Q(x))
    measurement_noise_dist(x) = MvNormal(zero(x), R(x))
    standard_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)
    omf_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)
    omf_invR_ensemble = rand(Xoshiro(42), init_dist, ENSEMBLE_SIZE)

    omf_cache = FilteringCache()
    omf_invR_cache = FilteringCache()
    standard_cache = FilteringCache()
    init_cache_ensemble!(omf_cache, omf_ensemble)
    init_cache_ensemble!(omf_invR_cache, omf_invR_ensemble)
    init_cache_ensemble!(standard_cache, standard_ensemble)

    standard_m = copy(μ₀)
    omf_m = copy(μ₀)
    omf_invR_m = copy(μ₀)
    standard_C = copy(Σ₀)
    omf_C = copy(Σ₀)
    omf_invR_C = copy(Σ₀)
    standard_traj = [(copy(μ₀), copy(Σ₀))]
    omf_traj = [(copy(μ₀), copy(Σ₀))]
    omf_invR_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        standard_ensemble = enkf_predict!(
            standard_cache,
            A(standard_m),
            process_noise_dist(standard_m),
            u(standard_m),
        )
        omf_ensemble = enkf_predict!(
            omf_cache,
            A(omf_m),
            process_noise_dist(omf_m),
            u(omf_m),
        )
        omf_invR_ensemble = enkf_predict!(
            omf_invR_cache,
            A(omf_invR_m),
            process_noise_dist(omf_invR_m),
            u(omf_invR_m),
        )

        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(copy(omf_ensemble))
        omf_invR_m, omf_invR_C = ensemble_mean_cov(copy(omf_invR_ensemble))

        standard_ensemble = enkf_correct!(
            standard_cache,
            H(standard_m),
            measurement_noise_dist(y),
            y,
            v(standard_m),
        )

        omf_centered_fens, omf_HX, omf_HA = PhDSE.A_HX_HA!(omf_cache, H(omf_m), v(omf_m))
        omf_ensemble = enkf_matrixfree_correct!(
            omf_cache,
            omf_HX,
            omf_HA,
            omf_centered_fens,
            measurement_noise_dist(y),
            y;
            R_inverse = missing,
        )

        omf_invR_centered_fens, omf_invR_HX, omf_invR_HA =
            PhDSE.A_HX_HA!(omf_invR_cache, H(omf_invR_m), v(omf_invR_m))
        omf_invR_ensemble = enkf_matrixfree_correct!(
            omf_invR_cache,
            omf_invR_HX,
            omf_invR_HA,
            omf_invR_centered_fens,
            measurement_noise_dist(y),
            y;
            R_inverse = inv(R(y)),
        )

        standard_m, standard_C = ensemble_mean_cov(standard_ensemble)
        omf_m, omf_C = ensemble_mean_cov(omf_ensemble)
        omf_invR_m, omf_invR_C = ensemble_mean_cov(omf_invR_ensemble)

        push!(standard_traj, (copy(standard_m), copy(standard_C)))
        push!(omf_traj, (copy(omf_m), copy(omf_C)))
        push!(omf_invR_traj, (copy(omf_invR_m), copy(omf_invR_C)))
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
    @test all([
        isapprox(m1, m2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(omf_traj, omf_invR_traj)
    ])
    @test all([
        isapprox(C1, C2; atol = 0.1, rtol = 0.1) for
        ((m1, C1), (m2, C2)) in zip(omf_traj, omf_invR_traj)
    ])

    if PLOT_RESULTS
        standard_means = [m for (m, C) in standard_traj]
        omf_means = [m for (m, C) in omf_traj]
        omf_invR_means = [m for (m, C) in omf_invR_traj]
        standard_stds = [2sqrt.(diag(C)) for (m, C) in standard_traj]
        omf_stds = [2sqrt.(diag(C)) for (m, C) in omf_traj]
        omf_invR_stds = [2sqrt.(diag(C)) for (m, C) in omf_invR_traj]
        using Plots
        test_plot1 =
            scatter(1:length(observations), [o[1] for o in observations], color = 1)
        plot!(
            test_plot1,
            1:length(ground_truth),
            [gt[1] for gt in ground_truth],
            label = "gt",
            color = :black,
            lw = 5,
            alpha = 0.4,
        )
        test_plot2 = plot(
            1:length(ground_truth),
            [gt[2] for gt in ground_truth],
            label = "gt",
            color = :black,
            lw = 5,
            alpha = 0.4,
        )
        plot!(
            test_plot1,
            1:length(standard_means),
            [m[1] for m in standard_means],
            ribbon = [s[1] for s in standard_stds],
            label = "standard",
            color = 5,
            lw = 3,
            ls = :dot,
        )
        plot!(
            test_plot2,
            1:length(standard_means),
            [m[2] for m in standard_means],
            ribbon = [s[2] for s in standard_stds],
            label = "standard",
            color = 5,
            lw = 3,
            ls = :dot,
        )
        plot!(
            test_plot1,
            1:length(omf_means),
            [m[1] for m in omf_means],
            ribbon = [s[1] for s in omf_stds],
            label = "omf",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(omf_means),
            [m[2] for m in omf_means],
            ribbon = [s[2] for s in omf_stds],
            label = "omf",
            color = 3,
            lw = 3,
        )
        plot!(
            test_plot1,
            1:length(omf_invR_means),
            [m[1] for m in omf_invR_means],
            ribbon = [s[1] for s in omf_invR_stds],
            label = "omf_invR",
            color = 4,
            ls = :dashdot,
            lw = 3,
        )
        plot!(
            test_plot2,
            1:length(omf_invR_means),
            [m[2] for m in omf_invR_means],
            ribbon = [s[2] for s in omf_invR_stds],
            label = "omf_invR",
            color = 4,
            ls = :dashdot,
            lw = 3,
        )
        test_plot = plot(test_plot1, test_plot2, layout = (1, 2))
        savefig(
            test_plot,
            joinpath(
                mkpath("./out/"),
                "standardEnKF_iip-vs-d3OMFEnKF_iip-vs-N3OMFEnKF_iip.png",
            ),
        )
    end
end
