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
    fcache::EnKFCache,
    Φ::AbstractMatrix,
    u::Union{AbstractVector,Missing} = missing,
)
    N = size(fcache.ensemble, 2)
    Distributions.rand!(fcache.process_noise_dist, fcache.forecast_ensemble)
    mul!(fcache.forecast_ensemble, Φ, fcache.ensemble, 1.0, 1.0)
    if !ismissing(u)
        fcache.forecast_ensemble .+= u
    end

    # Compute sample mean of forecast ensemble
    rdiv!(sum!(fcache.mX, fcache.forecast_ensemble), N)
    # Calculate zero-mean forecast ensemble by subtracting the mean
    copy!(fcache.A, fcache.forecast_ensemble)
    fcache.A .-= fcache.mX
end

"""
    enkf_correct!(fcache, H, R_inv, y, [v])

Correction step in an Ensemble Kalman filter (EnKF).

> **Note:**
>
> Calls [`omf_enkf_correct!`](@ref) intrinsically.

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
    mul!(fcache.HX, H, fcache.forecast_ensemble)  # [d, D] x [D, N] -> O(dDN)
    mul!(fcache.HA, H, fcache.A)  # [d, D] x [D, N] -> O(dDN)
    if !ismissing(v)
        fcache.HX .+= v
        fcache.HA .+= v
    end
    omf_enkf_correct!(fcache, R_inv, y)
end

"""
    omf_enkf_correct!(fcache, R_inv, y)

Correction step in an **o**bservation-**m**atrix-**f**ree (omf) Ensemble Kalman filter (EnKF),
assuming that the observation matrix never has to be built.

Instead, the products ``HX`` and ``HA = H\\cdot\\left(X^f - \\mathbb{E}\\left[X^f\\right]\\right)`` are pre-computed
_outside_ of the correction function and ``HX`` and ``HA`` are passed to the correction function.
Assuming `R_inv` is a `Diagonal` matrix and ``HX`` and ``HA`` are cheap to compute, this results in
a correction cost that is **linear** in the {state,observation}-dimension.

> **Note:**
>
> ``HX`` and ``HA`` have to be stored in the `fcache` before calling this function!

# Arguments
- `fcache::EnKFCache`: a cache holding memory-heavy objects
- `R_inv::AbstractMatrix`: *inverse of* the measurement noise covariance of the state space model
- `y::AbstractVector`: a measurement (data point)

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
"""
function omf_enkf_correct!(
    fcache::EnKFCache,
    R_inv::AbstractMatrix,
    y::AbstractVector,
)
    N = size(fcache.forecast_ensemble, 2)
    Nsub1 = N - 1

    # D - HX = ([y + v_i]_i=1:N) - HX , with v_i ~ N(0, R)
    Distributions.rand!(fcache.observation_noise_dist, fcache.dxN_cache01)
    fcache.dxN_cache01 .+= y
    fcache.dxN_cache01 .-= fcache.HX

    # R⁻¹ (D - HX)
    mul!(fcache.dxN_cache02, R_inv, fcache.dxN_cache01)

    # (HA)' R⁻¹(D - HX)
    mul!(fcache.NxN_cache01, fcache.HA', fcache.dxN_cache02)

    # Q := I_N + (HA)'R⁻¹(HA) / (N-1)
    #  -> R⁻¹(HA) / (N-1)
    rdiv!(mul!(fcache.dxN_cache03, R_inv, fcache.HA), Nsub1)
    #  -> (HA)'R⁻¹(HA) / (N-1)
    mul!(fcache.NxN_cache02, fcache.HA', fcache.dxN_cache03) # That's Q without the added Identity matrix
    #  -> I_N + (HA)'R⁻¹(HA) / (N-1)
    @inbounds @simd for i in 1:N
        fcache.NxN_cache02[i, i] += 1.0
    end # So that's Q now

    # Q⁻¹ (HA)' R⁻¹(D - HX)   /GETS OVERWRITTEN\   /GETS OVERWRITTEN\
    #                        |   by cholesky!   | |     by ldiv!     |
    ldiv!(cholesky!(Symmetric(fcache.NxN_cache02)), fcache.NxN_cache01)

    # K := (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
    rdiv!(mul!(fcache.dxN_cache02, fcache.HA, fcache.NxN_cache01), Nsub1)

    # (D - HX) - K
    fcache.dxN_cache01 .-= fcache.dxN_cache02

    # R⁻¹((D - HX) - K)
    mul!(fcache.dxN_cache04, R_inv, fcache.dxN_cache01)

    # (HA)' R⁻¹((D - HX) - K)
    mul!(fcache.NxN_cache02, fcache.HA', fcache.dxN_cache04)

    # Xᵃ = Xᶠ + A / (N-1) (HA)' R⁻¹((D - HX) - K)
    copy!(fcache.ensemble, fcache.forecast_ensemble)
    mul!(fcache.ensemble, rdiv!(fcache.A, Nsub1), fcache.NxN_cache02, 1.0, 1.0)
end

export enkf_predict!
export enkf_correct!
export omf_enkf_correct!
