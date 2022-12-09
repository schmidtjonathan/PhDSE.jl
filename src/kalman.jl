"""
    kf_predict(μ, Σ, Φ, Q, [u])

Prediction step in a Kalman filter.

For a numerically more stable version of the Kalman filter, have a look
at the square-root Kalman filter at [`sqrt_kf_predict`](@ref).

# Arguments
- `μ::AbstractVector`: the current filtering mean
- `Σ::AbstractMatrix`: the current filtering covariance matrix
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::AbstractMatrix`: transition covariance, i.e. process noise of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics
"""
function kf_predict(
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T},
    Φ::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    u::Union{AbstractVector{T},Missing} = missing,
) where {T}
    μ⁻ = Φ * μ
    if !ismissing(u)
        μ⁻ += u
    end
    Σ⁻ = Φ * Σ * Φ' + Q
    return μ⁻, Σ⁻
end


"""
    kf_correct(μ⁻, Σ⁻, H, R, y, [v])

Correction step in a Kalman filter.

For a numerically more stable version of the Kalman filter, either
have a look at the Joseph-form implementation [`kf_joseph_correct`](@ref) or
at the square-root Kalman filter at [`sqrt_kf_correct`](@ref).

# Arguments
- `μ⁻::AbstractVector`: the current predicted mean
- `Σ⁻::AbstractMatrix`: the current predicted covariance matrix
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R::AbstractMatrix`: measurement covariance, i.e. measurement noise of the state space model
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurements
"""
function kf_correct(
    μ⁻::AbstractVector{T},
    Σ⁻::AbstractMatrix{T},
    H::AbstractMatrix{T},
    R::AbstractMatrix{T},
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    ŷ = H * μ⁻
    if !ismissing(v)
        ŷ += v
    end
    cross_covariance = Σ⁻ * H'
    S = Symmetric(H * cross_covariance + R, :L)
    K = cross_covariance / S
    μ = μ⁻ + K * (y - ŷ)
    Σ = Σ⁻ - K * S * K'
    return μ, Σ
end



"""
    kf_joseph_correct(μ⁻, Σ⁻, H, R, y, [v])

Joseph-form correction step in a Kalman filter.

This method is numerically more stable than the standard [`kf_correct`](@ref) and
otherwise equivalent.

# Arguments
- `μ⁻::AbstractVector`: the current predicted mean
- `Σ⁻::AbstractMatrix`: the current predicted covariance matrix
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R::AbstractMatrix`: measurement covariance, i.e. measurement noise of the state space model
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurements
"""
function kf_joseph_correct(
    μ⁻::AbstractVector{T},
    Σ⁻::AbstractMatrix{T},
    H::AbstractMatrix{T},
    R::AbstractMatrix{T},
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    d, D = size(H)
    ŷ = H * μ⁻
    if !ismissing(v)
        ŷ += v
    end
    cross_covariance = Σ⁻ * H'
    S = Symmetric(H * cross_covariance + R, :L)
    K = cross_covariance / S
    μ = μ⁻ + K * (y - ŷ)
    I_KH = I(D) - K * H
    Σ = I_KH * Σ⁻ * I_KH' + K * R * K'
    return μ, Σ
end

export kf_predict
export kf_correct
export kf_joseph_correct

"""
    kf_predict!(c, Φ, Q, [u])

In-place prediction step in a Kalman filter.

For a numerically more stable version of the Kalman filter, have a look
at the square-root Kalman filter at [`sqrt_kf_predict!`](@ref).

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `Q::AbstractMatrix`: transition covariance, i.e. process noise of the state space model
- `u::AbstractVector` (optional): affine control input to the dynamics
"""
function kf_predict!(
    c::FilteringCache,
    Φ::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    u::Union{AbstractVector{T},Missing} = missing,
) where {T}
    D = size(Φ, 1)
    μ = get(c.entries, (Vector{T}, (D,), "mean")) do
        error("Cannot predict, no filtering mean in cache.")
    end
    Σ = get(c.entries, (Matrix{T}, (D, D), "covariance")) do
        error("Cannot predict, no filtering covariance in cache.")
    end
    μ⁻ = get!(
        c.entries,
        (Vector{T}, (D,), "predicted_mean"),
        similar(μ),
    )
    Σ⁻ = get!(
        c.entries,
        (Matrix{T}, (D, D), "predicted_covariance"),
        similar(Σ),
    )

    # predict mean
    # μ⁻ = Φμ [+ u]
    mul!(μ⁻, Φ, μ)
    if !ismissing(u)
        μ⁻ .+= u
    end

    # predict cov
    # Σ⁻ = ΦΣΦᵀ + Q
    ΣΦᵀ = get!(c.entries, (Matrix{T}, (D, D), "DxD_000"), similar(Σ))
    mul!(ΣΦᵀ, Σ, Φ')
    mul!(Σ⁻, Φ, ΣΦᵀ)
    Σ⁻ .+= Q
    return μ⁻, Σ⁻
end


"""
    kf_correct!(c, H, R, y, [v])

In-place correction step in a Kalman filter.

For a numerically more stable version of the Kalman filter,
have a look at the square-root Kalman filter at [`sqrt_kf_correct!`](@ref).

> [2022-12-09] There is currently no in-place implementation of the Joseph-form correction.

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `H::AbstractMatrix`: measurement matrix of the state space model
- `R::AbstractMatrix`: measurement covariance, i.e. measurement noise of the state space model
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurements
"""
function kf_correct!(
    c::FilteringCache,
    H::AbstractMatrix{T},
    R::AbstractMatrix{T},
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    d, D = size(H)
    μ⁻ = get(c.entries, (Vector{T}, (D,), "predicted_mean")) do
        error("Cannot correct, no predicted mean in cache.")
    end
    Σ⁻ = get(c.entries, (Matrix{T}, (D, D), "predicted_covariance")) do
        error("Cannot correct, no predicted covariance in cache.")
    end
    μ = get!(
        c.entries,
        (Vector{T}, (D,), "mean"),
        similar(μ⁻),
    )
    Σ = get!(
        c.entries,
        (Matrix{T}, (D, D), "covariance"),
        similar(Σ⁻),
    )

    # measure
    # ̂y = Hμ⁻ [+ v]
    ŷ = get!(c.entries, (Vector{T}, (d,), "d_000"), similar(y))
    mul!(ŷ, H, μ⁻)
    if !ismissing(v)
        ŷ .+= v
    end

    # compute update equations
    # Cross-covariance Σ⁻Hᵀ +++ [D, d]
    cross_cov = get!(c.entries, (Matrix{T}, (D, d), "Dxd_000"), Matrix{T}(undef, D, d))
    mul!(cross_cov, Σ⁻, H')
    # innovation matrix HΣ⁻Hᵀ + R
    S = get!(c.entries, (Matrix{T}, (d, d), "dxd_000"), Matrix{T}(undef, d, d))
    mul!(S, H, cross_cov)
    S .+= R

    # Kalman gain K = Σ⁻Hᵀ(HΣ⁻Hᵀ + R)⁻¹
    # The computation is (Σ⁻H') / S <=> Σ⁻H'S⁻¹ <=> Σ⁻H'(HΣ⁻H' + R)⁻¹
    # Note: cholesky! overwrites the lower-triangular part of S
    # Note 2: we reuse the memory of the cross-covariance, since it's not needed anymore
    K = cross_cov
    rdiv!(K, cholesky!(Symmetric(S, :L)))

    # μ = μ⁻ + K * (y - ŷ)
    residual = get!(c.entries, (Vector{T}, (d,), "d_001"), similar(y))
    copy!(residual, y)
    residual .-= ŷ
    mul!(μ, K, residual)
    μ .+= μ⁻

    # Σ = Σ⁻ - K * S * K' = Σ⁻ - (KL)(KL)' = Σ⁻ - KLL'K'
    copy!(Σ, Σ⁻)
    KLₛ = get!(c.entries, (Matrix{T}, (D, d), "Dxd_001"), similar(K))
    mul!(KLₛ, K, LowerTriangular(S))
    mul!(Σ, KLₛ, KLₛ', -1.0, 1.0)
    return μ, Σ
end

export kf_predict!
export kf_correct!
