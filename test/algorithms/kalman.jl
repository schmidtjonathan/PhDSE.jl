@testset "Kalman filter (IIP) vs. Kalman filter (OOP)" begin
    Random.seed!(1234)

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache()
    init_cache_moments!(cache, μ₀, Σ₀)
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "mean"))
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "predicted_mean"))
    @test haskey(cache.entries, (typeof(Σ₀), size(Σ₀), "covariance"))
    @test haskey(cache.entries, (typeof(Σ₀), size(Σ₀), "predicted_covariance"))

    iip_m = copy(μ₀)
    oop_m = copy(μ₀)
    iip_C = copy(Σ₀)
    oop_C = copy(Σ₀)
    iip_traj = [(copy(μ₀), copy(Σ₀))]
    oop_traj = [(copy(μ₀), copy(Σ₀))]
    for y in observations
        iip_m, iip_C = kf_predict!(cache, A, Q, u)
        oop_m, oop_C = kf_predict(oop_m, oop_C, A, Q, u)

        iip_m, iip_C = kf_correct!(cache, H, R, y, v)
        oop_m, oop_C = kf_correct(oop_m, oop_C, H, R, y, v)
        push!(iip_traj, (copy(iip_m), copy(iip_C)))
        push!(oop_traj, (copy(oop_m), copy(oop_C)))
    end

    @test all([m1 ≈ m2 for ((m1, C1), (m2, C2)) in zip(iip_traj, oop_traj)])
    @test all([C1 ≈ C2 for ((m1, C1), (m2, C2)) in zip(iip_traj, oop_traj)])

    if PLOT_RESULTS
        iip_means = stack([m for (m, C) in iip_traj])
        oop_means = stack([m for (m, C) in oop_traj])

        iip_stds = stack([2sqrt.(diag(C)) for (m, C) in iip_traj])
        oop_stds = stack([2sqrt.(diag(C)) for (m, C) in oop_traj])
        out_dir = mkpath("./out/kf_oop-vs-kf_iip")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=oop_means, estim_stds=oop_stds),
            joinpath(out_dir, "kf_oop.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=iip_means, estim_stds=iip_stds),
            joinpath(out_dir, "kf_iip.png")
        )
    end
    # for (k, v) in pairs(cache.entries)
    #     println("$k -> $(typeof(v)) of size $(size(v))")
    # end
end

@testset "Kalman filter (OOP) vs. Kalman filter (Joseph) (OOP)" begin
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
            kf_predict(standard_m, standard_C, A, Q, u)
        joseph_m, joseph_C =
            kf_predict(joseph_m, joseph_C, A, Q, u)

        standard_m, standard_C = kf_correct(
            standard_m,
            standard_C,
            H,
            R,
            y,
            v,
        )
        joseph_m, joseph_C =
            kf_joseph_correct(joseph_m, joseph_C, H, R, y, v)

        push!(standard_traj, (copy(standard_m), copy(standard_C)))
        push!(joseph_traj, (copy(joseph_m), copy(joseph_C)))
    end

    @test all([m1 ≈ m2 for ((m1, C1), (m2, C2)) in zip(standard_traj, joseph_traj)])
    @test all([C1 ≈ C2 for ((m1, C1), (m2, C2)) in zip(standard_traj, joseph_traj)])

    if PLOT_RESULTS
        standard_means = stack([m for (m, C) in standard_traj])
        joseph_means = stack([m for (m, C) in joseph_traj])

        standard_stds = stack([2sqrt.(diag(C)) for (m, C) in standard_traj])
        joseph_stds = stack([2sqrt.(diag(C)) for (m, C) in joseph_traj])
        out_dir = mkpath("./out/kf_joseph-vs-kf_standard")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=joseph_means, estim_stds=joseph_stds),
            joinpath(out_dir, "kf_joseph.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=standard_means, estim_stds=standard_stds),
            joinpath(out_dir, "kf_standard.png")
        )
    end
end
