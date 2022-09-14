"""
    enkf_predict!(fcache, Φ, Q, [u])

Prediction step in an Ensemble Kalman filter (EnKF).


# Arguments
- `fcache::EnKFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
"""
function enkf_predict!(
    fcache::Union{EnKFCache, EnKFCache2},
    Φ::AbstractMatrix,
    u::Union{AbstractVector,Missing} = missing,
)
    N = size(fcache.ensemble, 2)
    Distributions.rand!(fcache.process_noise_dist, fcache.forecast_ensemble)
    mul!(fcache.forecast_ensemble, Φ, fcache.ensemble, 1.0, 1.0)
    if !ismissing(u)
        fcache.forecast_ensemble .+= u
    end

    rdiv!(sum!(fcache.mX, fcache.forecast_ensemble), N)
    copy!(fcache.A, fcache.forecast_ensemble)
    fcache.A .-= fcache.mX
end

"""
    enkf_correct!(fcache, H, R_inv, y, [v])

Correction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `fcache::EnKFCache`: a cache holding memory-heavy objects
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R_inv::AbstractMatrix`: *inverse of* the measurement noise covariance of the state space model
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurement

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
"""
function enkf_correct!(
    fcache::EnKFCache,
    H::AbstractMatrix,
    R_inv::AbstractMatrix,
    y::AbstractVector,
    v::Union{AbstractVector,Missing} = missing,
)
    N = size(fcache.forecast_ensemble, 2)

    Distributions.rand!(fcache.observation_noise_dist, fcache.perturbed_D)
    fcache.perturbed_D .+= y

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



function enkf_correct!(
    fcache::EnKFCache2,
    HX::AbstractMatrix,
    HA::AbstractMatrix,
    R_inv::Diagonal,
    y::AbstractVector,
)
    N = size(fcache.forecast_ensemble, 2)
    Nsub1 = N - 1

    # D - HX = ([y + v_i]_i=1:N) - HX , with v_i ~ N(0, R)
    Distributions.rand!(fcache.observation_noise_dist, fcache.dxN_cache01)
    fcache.dxN_cache01 .+= y
    fcache.dxN_cache01 .-= HX

    # R⁻¹ (D - HX)
    mul!(fcache.dxN_cache02, R_inv, fcache.dxN_cache01)

    # (HA)' R⁻¹(D - HX)
    mul!(fcache.NxN_cache01, HA', fcache.dxN_cache02)

    # Q := I_N + (HA)'R⁻¹(HA) / (N-1)
    #  -> R⁻¹(HA) / (N-1)
    rdiv!(mul!(fcache.dxN_cache03, R_inv, HA), Nsub1)
    #  -> (HA)'R⁻¹(HA) / (N-1)
    mul!(fcache.NxN_cache02, HA', fcache.dxN_cache03) # That's Q without the added Identity matrix
    #  -> I_N + (HA)'R⁻¹(HA) / (N-1)
    @inbounds @simd for i in 1:N
        fcache.NxN_cache02[i, i] += 1.0
    end # So that's Q now

    # Q⁻¹ (HA)' R⁻¹(D - HX)                         /GETS OVERWRITTEN\
    ldiv!(cholesky!(Symmetric(fcache.NxN_cache02)), fcache.NxN_cache01)

    # K := (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
    rdiv!(mul!(fcache.dxN_cache02, HA, fcache.NxN_cache01), Nsub1)

    # (D - HX) - K
    fcache.dxN_cache01 .-= fcache.dxN_cache02

    # R⁻¹((D - HX) - K)
    mul!(fcache.dxN_cache04, R_inv, fcache.dxN_cache01)

    # (HA)' R⁻¹((D - HX) - K)
    mul!(fcache.NxN_cache02, HA', fcache.dxN_cache04)

    # Xᵃ = Xᶠ + A / (N-1) (HA)' R⁻¹((D - HX) - K)
    copy!(fcache.ensemble, fcache.forecast_ensemble)
    mul!(fcache.ensemble, rdiv!(fcache.A, Nsub1), fcache.NxN_cache02, 1.0, 1.0)
   end


export enkf_predict!
export enkf_correct!
