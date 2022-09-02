"""
    sqrt_kf_predict!(fcache, Φ, Q, [u])

Efficient and in-place implementation of the prediction step in a square-root Kalman filter.
Computes the same posterior distribution as the standard Kalman filter ([`kf_predict!`](@ref))
but is numerically more stable.
Works entirely on matrix-square-roots of the covariance matrices.

# Arguments
- `fcache::SqrtKFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::PSDMatrix`: **right** matrix square root of transition covariance, i.e. process noise of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics

# References
[1] Krämer, N., & Hennig, P. (2020). Stable implementation of probabilistic ODE solvers.
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
Computes the same posterior distribution as the standard Kalman filter ([`kf_correct!`](@ref))
but is numerically more stable.
Works entirely on matrix-square-roots of the covariance matrices.

# Arguments
- `fcache::SqrtKFCache`: a cache holding memory-heavy objects
- `y::AbstractVector`: a measurement (data point)
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R::PSDMatrix`: **right** matrix square root of measurement noise covariance of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement

# References
[1]:
Krämer, N., Bosch, N., Schmidt, J. & Hennig, P. (2022). Probabilistic ODE Solutions in Millions of Dimensions.
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
    # √Σ = R₂₂
    copy!(fcache.Σ.R, view(QR_R, d+1:d+D, d+1:d+D))
    # μ = μ⁻ + R₁₂ᵀ⋅ (R₁₁)⁻⋅(y - ̂y)
    mul!(
        fcache.μ,
        view(QR_R, 1:d, d+1:d+D)',
        ldiv!(LowerTriangular(view(QR_R, 1:d, 1:d)'), y .- fcache.obs_cache),
    )
    fcache.μ .+= fcache.μ⁻
end

export sqrt_kf_predict!
export sqrt_kf_correct!
