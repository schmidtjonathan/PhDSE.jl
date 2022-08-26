"""
    kf_predict!(fcache, Φ, [u], Q)

Efficient and in-place implementation of the prediction step in a Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics
- `Q::AbstractMatrix`: transition covariance, i.e. process noise of the state space model
"""
function kf_predict!(
    fcache::KFCache,
    Φ::AbstractMatrix,
    u::Union{AbstractVector,Missing},
    Q::AbstractMatrix,
)
    # predict mean
    # μ⁻ = Φμ [+ u]
    mul!(fcache.μ⁻, Φ, fcache.μ)
    if !ismissing(u)
        fcache.μ⁻ .+= u
    end

    # predict cov
    # Σ⁻ = ΦΣΦᵀ + Q
    mul!(fcache.predict_cache, fcache.Σ, Φ')
    mul!(fcache.Σ⁻, Φ, fcache.predict_cache)
    fcache.Σ⁻ .+= Q
end

"""
    kf_correct!(fcache, y, H, [v], R)

Efficient and in-place implementation of the correction step in a Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `y::AbstractVector`: a measurement (data point)
- `H::AbstractMatrix`: measurement matrix of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement
- `R::AbstractMatrix`: measurement noise covariance of the state space model
"""
function kf_correct!(
    fcache::KFCache,
    y::AbstractVector,
    H::AbstractMatrix,
    v::Union{AbstractVector,Missing},
    R::AbstractMatrix,
)

    # measure
    # ̂y = Hμ⁻ [+ v]
    mul!(fcache.obs_cache, H, fcache.μ⁻)
    if !ismissing(v)
        fcache.obs_cache .+= v
    end

    # compute update equations
    # Note: below we save the cross-covariance Σ⁻H' in the cache for the Kalman gain.
    # They have the same shape and the cross-covariance is used to compute the Kalman gain.
    mul!(fcache.K_cache, fcache.Σ⁻, H')
    mul!(fcache.S_cache, H, fcache.K_cache)
    fcache.S_cache .+= R
    # Note: here, the K_cache still holds the cross covariance.
    # The computation is (Σ⁻H') / S <=> Σ⁻H'S⁻¹ <=> Σ⁻H'(HΣ⁻H' + R)⁻¹
    rdiv!(fcache.K_cache, cholesky!(Symmetric(fcache.S_cache)))

    # μ = μ⁻ + K' * (y - ŷ)
    fcache.residual_cache .= y .- fcache.obs_cache
    mul!(fcache.μ, fcache.K_cache, fcache.residual_cache)
    fcache.μ .+= fcache.μ⁻

    # Σ = Σ⁻ - K * S * K'
    mul!(fcache.correct_cache, fcache.S_cache, fcache.K_cache')
    mul!(fcache.Σ, -fcache.K_cache, fcache.correct_cache)
    fcache.Σ .+= fcache.Σ⁻
end
