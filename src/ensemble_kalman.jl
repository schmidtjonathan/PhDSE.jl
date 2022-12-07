ensemble_mean(ens::AbstractMatrix) = vec(sum(ens, dims = 2)) / size(ens, 2)
centered_ensemble(ens::AbstractMatrix) = ens .- ensemble_mean(ens)
function ensemble_cov(ens::AbstractMatrix)
    A = centered_ensemble(ens)
    N_sub_1 = size(ens, 2) - 1
    return (A * A') / N_sub_1
end
function ensemble_mean_cov(ens::AbstractMatrix)
    m = ensemble_mean(ens)
    A = ens .- m
    N_sub_1 = size(ens, 2) - 1
    C = (A * A') / N_sub_1
    return m, C
end

export ensemble_mean
export centered_ensemble
export ensemble_cov
export ensemble_mean_cov

# --

function enkf_predict(
    ensemble::AbstractMatrix{T},
    Φ::AbstractMatrix{T},
    process_noise_dist::MvNormal,
    u::Union{AbstractVector{T},Missing} = missing,
) where {T}
    N = size(ensemble, 2)
    forecast_ensemble = Φ * ensemble + rand(process_noise_dist, N)
    if !ismissing(u)
        forecast_ensemble .+= u
    end

    # Compute sample mean of forecast ensemble
    # todo?
    # Calculate zero-mean forecast ensemble by subtracting the mean
    # todo?

    return forecast_ensemble
end

function enkf_correct(
    forecast_ensemble::AbstractMatrix{T},
    H::AbstractMatrix{T},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    N = size(forecast_ensemble, 2)
    HX = H * forecast_ensemble
    if !ismissing(v)
        HX .+= v
    end
    data_plus_noise = rand(measurement_noise_dist, N) .+ y
    residual = data_plus_noise - HX
    A = centered_ensemble(forecast_ensemble)

    C = Symmetric(A * A') / (N - 1)
    cross_covariance = C * H'
    Ŝ = Symmetric(H * cross_covariance + measurement_noise_dist.Σ, :L)
    K̂ = cross_covariance / Ŝ
    ensemble = forecast_ensemble + K̂ * residual
    return ensemble
end

function enkf_matrixfree_correct(
    forecast_ensemble::AbstractMatrix{T},
    HX::AbstractMatrix{T},
    HA::AbstractMatrix{T},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{T};
    A::Union{AbstractMatrix{T},Missing} = missing,
    R_inverse::Union{AbstractMatrix{T},Missing} = missing,
) where {T}
    D, N = size(forecast_ensemble)
    Nsub1 = N - 1
    d = length(y)
    data_plus_noise = rand(measurement_noise_dist, N) .+ y
    residual = data_plus_noise - HX
    if ismissing(A)
        A = centered_ensemble(forecast_ensemble)
    end

    compute_P_inverse = (d > N) && !ismissing(R_inverse)
    if compute_P_inverse
        @info "Matrix inversion lemma"
        # Implementation from the paper/preprint by Mandel; Section 4.2
        # uses the Matrix inversion lemma to be optimal for d >> N
        # -------------------------------------------------------------
        # R⁻¹ (D - HX)
        T1 = R_inverse * residual
        # (HA)' R⁻¹(D - HX)
        T2 = HA' * T1
        # Q := I_N + (HA)'(R⁻¹/ (N-1)) (HA)
        #  -> R⁻¹(HA) / (N-1)
        QT1 = (R_inverse / Nsub1) * HA
        #  -> (HA)'R⁻¹(HA) / (N-1)
        QT2 = HA' * QT1
        Q = Symmetric(I(D) + QT2, :L)
        # Q⁻¹ (HA)' R⁻¹(D - HX)
        T3 = Q \ T2
        # (1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
        T4 = Nsub1 \ (HA * T3)
        # (D - HX) - T4 (results from the identity matrix in [I - ...]; see Sec. 4.2
        T5 = residual - T4
        # P⁻¹(D - HX) = R⁻¹((D - HX) - ((1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)))

        P_inv_times_res = R_inverse * T5  # this is [d x d] * [d x N] -> [d x N]
    else
        # @info "Standard"
        # If N > d, use the standard approach instead, computing P,
        # instead of P⁻¹
        # @show measurement_noise_dist.Σ
        P = Nsub1 \ Symmetric(HA * HA' + measurement_noise_dist.Σ, :L)

        P_inv_times_res = P \ residual  # this is [d x d] * [d x N] -> [d x N]
    end
    # Now we insert P⁻¹(D - HX) into Eq. (2.2) ; see Sec. 2
    # (HA)' P⁻¹(D - HX)
    NxN_term = HA' * P_inv_times_res  # this is [N x d] * [d x N] -> [N x N]
    DxN_term = Nsub1 \ A * NxN_term  # this is [D x N] * [N x N] -> [N x N]

    ensemble = forecast_ensemble + DxN_term
    return ensemble
end

export enkf_predict
export enkf_correct
export enkf_matrixfree_correct

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

    # Q := I_N + (HA)'(R⁻¹/ (N-1)) (HA)
    #  -> R⁻¹(HA) / (N-1)
    mul!(fcache.dxN_cache03, rdiv!(R_inv, Nsub1), fcache.HA)
    #  -> (HA)'R⁻¹(HA) / (N-1)
    mul!(fcache.NxN_cache02, fcache.HA', fcache.dxN_cache03) # That's Q without the added Identity matrix
    #  -> I_N + (HA)'R⁻¹(HA) / (N-1)
    @inbounds @simd for i in 1:N
        fcache.NxN_cache02[i, i] += 1.0
    end # So that's Q now

    # Q⁻¹ (HA)' R⁻¹(D - HX)   /GETS OVERWRITTEN\   /GETS OVERWRITTEN\
    #                        |   by cholesky!   | |     by ldiv!     |
    ldiv!(cholesky!(Symmetric(fcache.NxN_cache02)), fcache.NxN_cache01)

    # K := (1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
    ldiv!(Nsub1, mul!(fcache.dxN_cache02, fcache.HA, fcache.NxN_cache01))

    # (D - HX) - K
    fcache.dxN_cache01 .-= fcache.dxN_cache02

    # R⁻¹((D - HX) - K)
    mul!(fcache.dxN_cache04, R_inv, fcache.dxN_cache01)

    # (HA)' R⁻¹((D - HX) - K)
    mul!(fcache.NxN_cache02, fcache.HA', fcache.dxN_cache04)

    # Xᵃ = Xᶠ + A / (N-1) (HA)' R⁻¹((D - HX) - K)
    copy!(fcache.ensemble, fcache.forecast_ensemble)
    mul!(fcache.ensemble, ldiv!(Nsub1, fcache.A), fcache.NxN_cache02, 1.0, 1.0)
end

export enkf_predict!
export enkf_correct!
export omf_enkf_correct!
