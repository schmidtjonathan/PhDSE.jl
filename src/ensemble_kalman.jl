function enkf_predict!(fcache::EnKFCache, Φ, u = missing)
    Distributions.rand!(fcache.process_noise_dist, fcache.forecast_ensemble)
    mul!(fcache.forecast_ensemble, Φ, fcache.ensemble, 1.0, 1.0)
    if !ismissing(u)
        fcache.forecast_ensemble .+= u
    end
end

function enkf_correct!(fcache::EnKFCache, H, R_inv, y, v = missing)
    D, N = size(fcache.forecast_ensemble)

    Distributions.rand!(fcache.observation_noise_dist, fcache.perturbed_D)
    fcache.perturbed_D .+= y

    rdiv!(sum!(fcache.mX, fcache.forecast_ensemble), N) # mean
    copy!(fcache.A, fcache.forecast_ensemble)
    fcache.A .-= fcache.mX

    mul!(fcache.HX, H, fcache.forecast_ensemble)  # [d, D] x [D, N] -> O(dDN)
    mul!(fcache.HA, H, fcache.A)  # [d, D] x [D, N] -> O(dDN)
    if !ismissing(v)
        fcache.HX .+= v
        fcache.HA .+= v
    end

    mul!(fcache.HAt_x_Rinv, fcache.HA', R_inv)  # [N, d] x [d, d] -> O(Nd^2)
    rdiv!(mul!(fcache.Q, fcache.HAt_x_Rinv, fcache.HA), N - 1)  # [N, d] x [d, N] -> # [d, D] x [D, N] -> O(dN²)
    # the loop adds a identity matrix
    @inbounds for i in axes(fcache.Q, 1)
        fcache.Q[i, i] += 1.0
    end
    ldiv!(fcache.W, cholesky!(Symmetric(fcache.Q)), fcache.HAt_x_Rinv)  # [N, N] \ [N, d] -> O(N²d)
    rdiv!(mul!(fcache.M, fcache.HA, fcache.W), 1 - N)  # [d, N] x [N, d] -> O(d²N)
    # the loop calculates I - M
    @inbounds for i in axes(fcache.M, 1)
        fcache.M[i, i] += 1.0
    end
    mul!(fcache.HAt_x_S_inv, fcache.HAt_x_Rinv, fcache.M)  # [N, d] x [d, d] -> O(Nd²)
    mul!(fcache.AHAt_x_Sinv, fcache.A, fcache.HAt_x_S_inv)  # [D, N] x [N, d] -> O(DNd)
    fcache.residual .= fcache.perturbed_D .- fcache.HX

    copy!(fcache.ensemble, fcache.forecast_ensemble)
    mul!(fcache.ensemble, fcache.AHAt_x_Sinv, fcache.residual, inv(N - 1), 1.0)  # [D, d] x [d, N] -> O(DdN)
end

export enkf_predict!
export enkf_correct!
