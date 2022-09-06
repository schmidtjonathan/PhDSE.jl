function ensemble_mean(X)
    D, N = size(X)
    e_N1 = ones(N, 1)
    return (X * e_N1) ./ N
end


function ensemble_cov(X)
    D, N = size(X)
    A = X - ensemble_mean(X) * ones(1, N)
    return A * A' / (N - 1)
end


function enkf_predict!(fcache::EnKFCache, Φ, u = missing)

    Distributions.rand!(fcache.process_noise_dist, fcache.forecast_ensemble)
    @simd for i in axes(fcache.forecast_ensemble, 2)
        fcache.forecast_ensemble[:, i] .+= Φ * fcache.ensemble[:, i]
        if !ismissing(u)
            fcache.forecast_ensemble[:, i] .+= u
        end
    end
end



function enkf_correct!(fcache::EnKFCache, H, R_inv, y, v = missing)
    D, N = size(fcache.forecast_ensemble)
    d = size(H, 1)

    by_N = 1.0 / N
    by_Nm1 = 1.0 / (N - 1)
    e_N1 = ones(N, 1)
    e_1N = ones(1, N)

    perturbed_D = y .+ rand(fcache.observation_noise_dist, N)

    X = fcache.forecast_ensemble
    A = X - by_N * X * e_N1 * e_1N
    HA = H * A
    Q = I(N) + HA' * R_inv * by_Nm1 * HA
    S_inv = R_inv * (I(d) - by_Nm1 * HA * (cholesky(Symmetric(Q)) \ (HA' * R_inv)))
    m_new = X + by_Nm1 * A * HA' * S_inv * (perturbed_D - H * X)
    copy!(fcache.ensemble, m_new)
end