function enkf_predict!(fcache::EnKFCache, Φ, Q, chol_Q, u = missing)

    mul!(fcache.forecast_ensemble, Φ, fcache.ensemble)
    if !ismissing(u)
        fcache.forecast_ensemble .+= u
    end

    # ToDo:  ALLOCATES -----------------------v
    mul!(fcache.forecast_perturb, chol_Q.L, randn(size(fcache.forecast_ensemble)))
    fcache.forecast_ensemble .+= fcache.forecast_perturb
end



function enkf_correct!(fcache::EnKFCache, H, R, chol_R, y, v = missing)
    D, N = size(fcache.forecast_ensemble)

    HX = H * fcache.forecast_ensemble
    z = HX * ones(N, 1)
    HA = HX .- ((1.0/N) .* z * ones(1, N))
    D = y .+ chol_R.L * randn(length(y), N)
    Y = D - HX
    P = R + ((1.0 / (N - 1)) .* HA * HA'
    chol_P = cholesky(Symmetric(P))
    M =
end