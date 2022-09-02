"""
    sqrt_kf_predict!(fcache, Φ, Q, [u])

Efficient and in-place implementation of the prediction step in a square-root Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::PSDMatrix`: **right** matrix square root of transition covariance, i.e. process noise of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics
"""
function sqrt_kf_predict!(
    fcache::SqrtKFCache,
    Φ::AbstractMatrix,
    Q::PSDMatrix,
    u::Union{AbstractVector,Missing} = missing,
)
    # predict mean
    mul!(fcache.μ⁻, Φ, fcache.μ)
    if !ismissing(u)
        fcache.μ⁻ .+= u
    end

    # predict cov
    D = size(Q, 1)

    mul!(view(fcache.cache_2DxD, 1:D, 1:D), fcache.Σ.R, Φ')
    copy!(view(fcache.cache_2DxD, D+1:2D, 1:D), Q.R)
    copy!(fcache.Σ⁻.R, qr!(fcache.cache_2DxD).R)
end

"""
    sqrt_kf_correct!(fcache, H, R, y, [v])

Efficient and in-place implementation of the correction step in a square-root Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `y::AbstractVector`: a measurement (data point)
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R::PSDMatrix`: **right** matrix square root of measurement noise covariance of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement
"""
function sqrt_kf_correct!(
    fcache::SqrtKFCache,
    H::AbstractMatrix,
    R::PSDMatrix,
    y::AbstractVector,
    v::Union{AbstractVector,Missing} = missing,
)

    # measure
    # ̂y = Hμ⁻ [+ v]
    mul!(fcache.obs_cache, H, fcache.μ⁻)
    if !ismissing(v)
        fcache.obs_cache .+= v
    end

    d, D = size(H)

    # begin  (Eq. (31) in highdim paper) >>>
    # Populate big block matrix
    # top left: sqrt(Σ⁻) * H'
    mul!(view(fcache.cache_dpDxdpD, 1:D, 1:d), fcache.Σ⁻.R, H')
    # top right: sqrt(Σ⁻)
    copy!(view(fcache.cache_dpDxdpD, 1:D, d+1:d+D), fcache.Σ⁻.R)
    # bottom left: sqrt(R)
    copy!(view(fcache.cache_dpDxdpD, D+1:D+d, 1:d), R.R)
    # bottom right: 0_dxD
    copy!(view(fcache.cache_dpDxdpD, D+1:D+d, d+1:d+D), fcache.zero_cache_dxD)

    # QR-decompose
    QR_R = qr!(fcache.cache_dpDxdpD).R

    # Read out relevant matrices
    # R_22
    copy!(fcache.Σ.R, view(QR_R, d+1:d+D, d+1:d+D))

    # R_11
    copy!(fcache.S_cache.R, view(QR_R, 1:d, 1:d))

    # R_12ᵀ * R_11 (cross-covariance, save in K-cache for now)
    mul!(fcache.K_cache, view(QR_R, 1:d, d+1:d+D)', fcache.S_cache.R)

    # R_12ᵀ S⁻¹
    rdiv!(fcache.K_cache, Cholesky(fcache.S_cache.R, :U, 0))

    # <<< end  (Eq. (31) in highdim paper)

    fcache.residual_cache .= y .- fcache.obs_cache
    mul!(fcache.μ, fcache.K_cache, fcache.residual_cache)
    fcache.μ .+= fcache.μ⁻
end

export sqrt_kf_predict!
export sqrt_kf_correct!