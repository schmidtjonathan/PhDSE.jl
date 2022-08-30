"""
    sqrt_kf_predict!(fcache, Φ, [u], Q)

Efficient and in-place implementation of the prediction step in a square-root Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::PSDMatrix`: transition covariance, i.e. process noise of the state space model
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
    R, M = fcache.cache_2DxD, fcache.cache_DxD
    D = size(Q, 1)

    mul!(view(R, 1:D, 1:D), fcache.Σ.R, Φ')
    view(R, D+1:2D, 1:D) .= Q.R
    mul!(M, R', R)
    chol = cholesky!(Symmetric(M, :U), check=false)
    copy!(fcache.Σ⁻.R, issuccess(chol) ? Matrix(chol.U) : qr!(R).R)
end

"""
    sqrt_kf_correct!(fcache, y, H, [v], R)

Efficient and in-place implementation of the correction step in a square-root Kalman filter.

# Arguments
- `fcache::KFCache`: a cache holding memory-heavy objects
- `y::AbstractVector`: a measurement (data point)
- `H::AbstractMatrix`: measurement matrix of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement
- `R::AbstractMatrix`: measurement noise covariance of the state space model
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

    D = size(R, 1)

    copy!(fcache.S_cache, R)
    S_chol = Cholesky(qr!(fcache.S_cache.R).R, :U, 0)
    mul!(fcache.K_cache, Matrix(fcache.Σ⁻), H')
    rdiv!(fcache.K_cache, S_chol)

    fcache.residual_cache .= y .- fcache.obs_cache
    mul!(fcache.μ, fcache.K_cache, fcache.residual_cache)
    fcache.μ .+= fcache.μ⁻

    mul!(fcache.cache_DxD, fcache.K_cache, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        fcache.cache_DxD[i, i] += 1
    end

    X_A_Xt!(fcache.Σ, fcache.Σ⁻, fcache.cache_DxD)

end

export sqrt_kf_predict!
export sqrt_kf_correct!
