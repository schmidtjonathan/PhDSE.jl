@testset "Kalman filter (OOP) vs. SqrtKF (IIP) vs. SqrtKF (OOP)" begin
    Random.seed!(1234)
    upper_sqrt_to_mat(MU::UpperTriangular) = MU' * MU

    μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations = filtering_setup()
    cache = FilteringCache()
    init_cache_moments!(cache, μ₀, cholesky(Σ₀).U)
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "mean"))
    @test haskey(cache.entries, (typeof(μ₀), size(μ₀), "predicted_mean"))

    kf_m = copy(μ₀)
    iip_sqrt_kf_m = copy(μ₀)
    oop_sqrt_kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    iip_sqrt_kf_C = cholesky(Σ₀).U
    oop_sqrt_kf_C = cholesky(Σ₀).U

    @test haskey(cache.entries, (typeof(oop_sqrt_kf_C), size(oop_sqrt_kf_C), "covariance"))
    @test haskey(
        cache.entries,
        (typeof(oop_sqrt_kf_C), size(oop_sqrt_kf_C), "predicted_covariance"),
    )

    kf_traj = [(copy(μ₀), copy(Σ₀))]
    iip_traj = [(copy(μ₀), copy(Σ₀))]
    oop_traj = [(copy(μ₀), copy(Σ₀))]
    sqrt_Q = UpperTriangular(cholesky(Q).U)
    sqrt_R = UpperTriangular(cholesky(R).U)
    for y in observations
        kf_m, kf_C = kf_predict(kf_m, kf_C, A, Q, u)
        iip_sqrt_kf_m, iip_sqrt_kf_C = sqrt_kf_predict!(
            cache,
            A,
            sqrt_Q,
            u,
        )
        oop_sqrt_kf_m, oop_sqrt_kf_C = sqrt_kf_predict(
            oop_sqrt_kf_m,
            oop_sqrt_kf_C,
            A,
            sqrt_Q,
            u,
        )

        kf_m, kf_C = kf_correct(kf_m, kf_C, H, R, y, v)
        iip_sqrt_kf_m, iip_sqrt_kf_C = sqrt_kf_correct!(
            cache,
            H,
            sqrt_R,
            y,
            v,
        )
        oop_sqrt_kf_m, oop_sqrt_kf_C = sqrt_kf_correct(
            oop_sqrt_kf_m,
            oop_sqrt_kf_C,
            H,
            sqrt_R,
            y,
            v,
        )

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
        kf_means = stack([m for (m, C) in kf_traj])
        iip_means = stack([m for (m, C) in iip_traj])
        oop_means = stack([m for (m, C) in oop_traj])
        kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj])
        iip_stds = stack([2sqrt.(diag(C)) for (m, C) in iip_traj])
        oop_stds = stack([2sqrt.(diag(C)) for (m, C) in oop_traj])

        out_dir = mkpath("./out/kf_oop-vs-sqrtkf_iip-vs-sqrtkf_oop")
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=kf_means, estim_stds=kf_stds),
            joinpath(out_dir, "kf.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=iip_means, estim_stds=iip_stds),
            joinpath(out_dir, "sqrtkf_iip.png")
        )
        savefig(
            plot_test(stack(ground_truth), stack(observations), H; estim_means=oop_means, estim_stds=oop_stds),
            joinpath(out_dir, "sqrtkf_oop.png")
        )
    end
    # for (k, v) in pairs(cache.entries)
    #     println("$k -> $(typeof(v)) of size $(size(v))")
    # end
end
