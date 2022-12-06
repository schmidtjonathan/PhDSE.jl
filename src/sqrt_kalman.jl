function sqrt_kf_predict(
    μ::AbstractVector{T},
    sqrt_Σ::UpperTriangular{T,PT},
    Φ::AbstractMatrix{T},
    sqrt_Q::UpperTriangular{T,PT},
    u::Union{AbstractVector{T},Missing} = missing,
) where {T,PT<:AbstractMatrix}
    μ⁻ = Φ * μ
    if !ismissing(u)
        μ⁻ += u
    end
    sqrt_Σ⁻ = UpperTriangular(qr([sqrt_Σ * Φ'; sqrt_Q]).R)
    return μ⁻, sqrt_Σ⁻
end

function sqrt_kf_correct(
    μ⁻::AbstractVector{T},
    sqrt_Σ⁻::UpperTriangular{T,PT},
    H::AbstractMatrix{T},
    sqrt_R::UpperTriangular{T,PT},
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T,PT<:AbstractMatrix}
    d, D = size(H)
    ŷ = H * μ⁻
    if !ismissing(v)
        ŷ += v
    end
    X = qr(
        [sqrt_Σ⁻*H' sqrt_Σ⁻
            sqrt_R zero(H)],
    ).R
    R₂₂ = X[d+1:end, d+1:end]
    R₁₂ = X[1:d, d+1:end]
    R₁₁ = X[1:d, 1:d]
    μ = μ⁻ + R₁₂' * (LowerTriangular(R₁₁') \ (y - ŷ))
    sqrt_Σ = UpperTriangular(R₂₂)
    return μ, sqrt_Σ
end

export sqrt_kf_predict
export sqrt_kf_correct

"""
    sqrt_kf_predict!(fcache, Φ, Q, [u])

Efficient and in-place implementation of the prediction step in a square-root Kalman filter.
Computes the same posterior distribution as the standard Kalman filter ([`kf_predict!`](@ref))
but is numerically more stable.
Works entirely on matrix-square-roots of the covariance matrices.

# Arguments
- `fcache::SqrtKFCache`: a cache holding memory-heavy objects
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::RightMatrixSquareRoot`: **right** matrix square root of transition covariance, i.e. process noise of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics

# References
[1] Krämer, N., & Hennig, P. (2020). Stable implementation of probabilistic ODE solvers.
"""
function sqrt_kf_predict!(
    c::FilteringCache,
    Φ::AbstractMatrix{T},
    sqrt_Q::UpperTriangular{T,PT},
    u::Union{AbstractVector{T},Missing} = missing,
) where {T,PT<:AbstractMatrix}
    D = size(Φ, 1)
    μ = get(c.entries, (Vector{T}, (D,), "mean")) do
        error("Cannot predict, no filtering mean in cache.")
    end
    sqrt_Σ = get(c.entries, (UpperTriangular{T,PT}, (D, D), "covariance")) do
        error("Cannot predict, no filtering covariance in cache.")
    end
    μ⁻ = get!(
        c.entries,
        (Vector{T}, (D,), "predicted_mean"),
        similar(μ),
    )
    sqrt_Σ⁻ = get!(
        c.entries,
        (UpperTriangular{T,PT}, (D, D), "predicted_covariance"),
        similar(sqrt_Σ),
    )

    # predict mean
    mul!(μ⁻, Φ, μ)
    if !ismissing(u)
        μ⁻ .+= u
    end

    # predict cov
    cache_2DxD = get!(c.entries, (Matrix{T}, (2D, D), "2DxD_000"), Matrix{T}(undef, 2D, D))
    mul!(view(cache_2DxD, 1:D, 1:D), sqrt_Σ, Φ')
    copy!(view(cache_2DxD, D+1:2D, 1:D), sqrt_Q)
    copy!(sqrt_Σ⁻, qr!(cache_2DxD).R)
    return μ⁻, sqrt_Σ⁻
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
- `R::RightMatrixSquareRoot`: **right** matrix square root of measurement noise covariance of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement

# References
[1]:
Krämer, N., Bosch, N., Schmidt, J. & Hennig, P. (2022). Probabilistic ODE Solutions in Millions of Dimensions.
"""
function sqrt_kf_correct!(
    c::FilteringCache,
    H::AbstractMatrix{T},
    sqrt_R::UpperTriangular{T,PT},
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T,PT<:AbstractMatrix}
    d, D = size(H)
    μ⁻ = get(c.entries, (Vector{T}, (D,), "predicted_mean")) do
        error("Cannot correct, no predicted mean in cache.")
    end
    sqrt_Σ⁻ =
        get(c.entries, (UpperTriangular{T,PT}, (D, D), "predicted_covariance")) do
            error("Cannot correct, no predicted covariance in cache.")
        end
    μ = get!(
        c.entries,
        (Vector{T}, (D,), "mean"),
        similar(μ⁻),
    )
    sqrt_Σ = get!(
        c.entries,
        (UpperTriangular{T,PT}, (D, D), "covariance"),
        similar(sqrt_Σ⁻),
    )

    # measure
    # ̂y = Hμ⁻ [+ v]
    ŷ = get!(c.entries, (Vector{T}, (d,), "d_000"), similar(y))
    mul!(ŷ, H, μ⁻)
    if !ismissing(v)
        ŷ .+= v
    end

    # Populate big block matrix
    cache_dpDxdpD = get!(
        c.entries,
        (Matrix{T}, (d + D, d + D), "d+Dxd+D_000"),
        Matrix{T}(undef, d + D, d + D),
    )
    dxD_zero_mat = get!(c.entries, (Matrix{T}, (d, D), "dxD_zeros"), zero(H))
    # top left: sqrt(Σ⁻) * H'
    mul!(view(cache_dpDxdpD, 1:D, 1:d), sqrt_Σ⁻, H')
    # top right: sqrt(Σ⁻)
    copy!(view(cache_dpDxdpD, 1:D, d+1:d+D), sqrt_Σ⁻)
    # bottom left: sqrt(R)
    copy!(view(cache_dpDxdpD, D+1:D+d, 1:d), sqrt_R)
    # bottom right: 0_dxD
    copy!(view(cache_dpDxdpD, D+1:D+d, d+1:d+D), dxD_zero_mat)

    # QR-decompose
    QR_R = qr!(cache_dpDxdpD).R

    # Read out relevant matrices
    # √Σ = R₂₂
    copy!(sqrt_Σ, view(QR_R, d+1:d+D, d+1:d+D))
    # μ = μ⁻ + R₁₂ᵀ⋅ (R₁₁)⁻⋅(y - ̂y)
    residual = get!(c.entries, (Vector{T}, (d,), "d_001"), similar(y))
    copy!(residual, y)
    residual .-= ŷ
    mul!(
        μ,
        view(QR_R, 1:d, d+1:d+D)',
        ldiv!(LowerTriangular(view(QR_R, 1:d, 1:d)'), residual),
    )
    μ .+= μ⁻
    return μ, sqrt_Σ
end

export sqrt_kf_predict!
export sqrt_kf_correct!
