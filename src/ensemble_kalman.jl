"""
    ensemble_mean(ens)

Compute the mean ``\\bar{\\mathrm{X}}`` of an ensemble of states ``\\mathrm{X} \\in \\mathbb{R}^{D \\times N}``
"""
ensemble_mean(ens::AbstractMatrix) = vec(sum(ens, dims = 2)) / size(ens, 2)

"""
    centered_ensemble(ens)

Compute the mean-centered ensemble ``\\mathrm{A} = \\mathrm{X} - \\bar{\\mathrm{X}}``
"""
centered_ensemble(ens::AbstractMatrix) = ens .- ensemble_mean(ens)

"""
    ensemble_cov(ens)

Compute the sample covariance of an ensemble ``\\tilde{\\mathrm{C}} = \\frac{\\mathrm{A}\\mathrm{A}^\\top}{N - 1}``
"""
function ensemble_cov(ens::AbstractMatrix)
    A = centered_ensemble(ens)
    N_sub_1 = size(ens, 2) - 1
    return (A * A') / N_sub_1
end

"""
    ensemble_mean_cov(ens)

Compute sample moments of an ensemble as in [`ensemble_mean`](@ref) and [`ensemble_cov`](@ref).
"""
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

"""
    ensemble_mean!(out_m, ens)

Compute the mean ``\\bar{\\mathrm{X}}`` of an ensemble of states ``\\mathrm{X} \\in \\mathbb{R}^{D \\times N}``
"""
function ensemble_mean!(out_m::AbstractVector, ens::AbstractMatrix)
    rdiv!(sum!(out_m, ens), size(ens, 2))
    return out_m
end

"""
    centered_ensemble!(out_ens, out_m, ens)

Compute the mean-centered ensemble ``\\mathrm{A} = \\mathrm{X} - \\bar{\\mathrm{X}}``
"""
function centered_ensemble!(
    out_ens::AbstractMatrix,
    out_m::AbstractVector,
    ens::AbstractMatrix,
)
    out_m = ensemble_mean!(out_m, ens)
    copy!(out_ens, ens)
    out_ens .-= out_m
    return out_ens
end

"""
    ensemble_mean_cov!(out_cov, out_ens, out_m, ens)

Compute sample moments of an ensemble. See [`ensemble_mean_cov`](@ref).
"""
function ensemble_mean_cov!(
    out_cov::AbstractMatrix,
    out_ens::AbstractMatrix,
    out_m::AbstractVector,
    ens::AbstractMatrix,
)
    A = centered_ensemble!(out_ens, out_m, ens)
    N_sub_1 = size(ens, 2) - 1
    rdiv!(mul!(out_cov, A, A'), N_sub_1)
    return out_m, out_cov
end

export ensemble_mean!
export centered_ensemble!
export ensemble_mean_cov!

# --

"""
    enkf_predict(ensemble, Φ, process_noise_dist, [u])

Prediction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `ensemble::AbstractMatrix`: the current analysis ensemble
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `process_noise_dist::MvNormal`: Multivariate Gaussian process-noise distribution.
- `u::AbstractVector` (optional): affine control input to the dynamics

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
"""
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

    return forecast_ensemble
end

# >>>
# Houtekamer, Mitchell, 1998. "Data Assimilation Using an Ensemble Kalman Filter Technique".
# Eqs. (13) - (15)
function _sum_terms(ens_member, ens_mean, H)
    centered = ens_member - ens_mean
    meas = H * centered
    PH_term = centered * meas'
    HPH_term = meas * meas'
    return PH_term, HPH_term
end

function _calc_PH_HPH(ensemble, H)
    D, N = size(ensemble)
    d = size(H, 1)
    ens_mean = ensemble_mean(ensemble)
    PH = zeros(D, d)
    HPH = zeros(d, d)
    @inbounds @simd for i in 1:N
        PH_term, HPH_term = _sum_terms(ensemble[:, i], ens_mean, H)
        PH .+= PH_term
        HPH .+= HPH_term
    end
    return PH / (N - 1), HPH / (N - 1)
end

# <<<

"""
    enkf_correct(forecast_ensemble, H, measurement_noise_dist, y, [v])

Correction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `forecast_ensemble::AbstractMatrix`: the current forecast ensemble
- `H::AbstractMatrix`: measurement matrix of the state space model
- `measurement_noise_dist::MvNormal`: Multivariate Gaussian measurement-noise distribution.
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurement

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter. Section 2.
"""
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
    PH, HPH = _calc_PH_HPH(forecast_ensemble, H)

    Ŝ = HPH + measurement_noise_dist.Σ
    K̂ = PH / cholesky!(Symmetric(Ŝ))
    ensemble = forecast_ensemble + K̂ * residual
    return ensemble
end

"""
    enkf_matrixfree_correct(forecast_ensemble, HX, HA, measurement_noise_dist, y; [A], [R_inverse])

Observation-matrix-free correction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `forecast_ensemble::AbstractMatrix`: the current forecast ensemble
- `HX::AbstractMatrix`: measured forecast ensemble
- `HA::AbstractMatrix`: measured and centered forecast ensemble
- `measurement_noise_dist::MvNormal`: Multivariate Gaussian measurement-noise distribution.
- `y::AbstractVector`: a measurement (data point)
- `A::AbstractMatrix` (optional): centered forecast ensemble
- `R_inverse::AbstractMatrix` (optional): the inverse ``R^{-1}`` of the measurement covariance matrix.
    `missing` per default.
    If `R_inverse` is not `missing`, then an alternative computation of the analysis ensemble is chosen, which uses
    the matrix-inversion lemma to scale as `O(N^3)`, instead of `O(d^3)`!

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter. Sections 3, 4.
"""
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

    compute_P_inverse = !ismissing(R_inverse)
    if compute_P_inverse
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
        Q = Symmetric(I(N) + QT2, :L)
        # Q⁻¹ (HA)' R⁻¹(D - HX)
        T3 = Q \ T2
        # (1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
        T4 = Nsub1 \ (HA * T3)
        # (D - HX) - T4 (results from the identity matrix in [I - ...]; see Sec. 4.2
        T5 = residual - T4
        # P⁻¹(D - HX) = R⁻¹((D - HX) - ((1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)))

        P_inv_times_res = R_inverse * T5  # this is [d x d] * [d x N] -> [d x N]
    else
        # If N > d, use the standard approach instead, computing P,
        # instead of P⁻¹
        P = Symmetric(Nsub1 \ HA * HA' + measurement_noise_dist.Σ, :L)

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
    enkf_predict!(c, Φ, process_noise_dist, [u])

In-place prediction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `Φ::AbstractMatrix`: transition matrix, i.e. dynamics of the state space model
- `process_noise_dist::MvNormal`: Multivariate Gaussian process-noise distribution.
- `u::AbstractVector` (optional): affine control input to the dynamics

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
"""
function enkf_predict!(
    c::FilteringCache,
    Φ::AbstractMatrix{T},
    process_noise_dist::MvNormal,
    u::Union{AbstractVector{T},Missing} = missing,
) where {T}
    D = size(Φ, 1)
    N = get(c.entries, (typeof(D), size(D), "N")) do
        error("Ensemble size N is missing in FilteringCache.")
    end
    ensemble = get(c.entries, (Matrix{T}, (D, N), "ensemble")) do
        error("Cannot predict, no analysis ensemble in cache.")
    end
    forecast_ensemble = get!(
        c.entries,
        (typeof(ensemble), size(ensemble), "forecast_ensemble"),
        similar(ensemble),
    )

    Distributions.rand!(process_noise_dist, forecast_ensemble)
    mul!(forecast_ensemble, Φ, ensemble, 1.0, 1.0)
    if !ismissing(u)
        forecast_ensemble .+= u
    end

    return forecast_ensemble
end

"""
    A_HX_HA!(c, H, [v])

In-place computation of measured forecast ensemble `HX`,
as well as measured centered forecast ensemble `HA` and centered forecast ensemble `A`.

> Note: not `export`ed. Use as `PhDSE.A_HX_HA!`.

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `H::AbstractMatrix`: measurement matrix of the state space model
- `v::AbstractVector` (optional): affine control input to the measurement
"""
function A_HX_HA!(
    c::FilteringCache,
    H::AbstractMatrix{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    d, D = size(H)
    N = get(c.entries, (typeof(D), size(D), "N")) do
        error("Ensemble size N is missing in FilteringCache.")
    end
    forecast_ensemble = get(c.entries, (Matrix{T}, (D, N), "forecast_ensemble")) do
        error("Cannot compute HX/HA, no forecast ensemble in cache.")
    end
    HX = get!(c.entries, (Matrix{T}, (d, N), "HX"), Matrix{T}(undef, d, N))
    HA = get!(c.entries, (typeof(HX), size(HX), "HA"), similar(HX))
    mul!(HX, H, forecast_ensemble)
    if !ismissing(v)
        HX .+= v
    end

    ens_mean =
        get!(c.entries, (Vector{T}, (D,), "forecast_ensemble_mean"), Vector{T}(undef, D))
    A = get!(
        c.entries,
        (typeof(forecast_ensemble), size(forecast_ensemble), "centered_forecast_ensemble"),
        similar(forecast_ensemble),
    )
    centered_ensemble!(A, ens_mean, forecast_ensemble)

    measured_ens_mean =
        get!(
            c.entries,
            (Vector{T}, (d,), "measured_forecast_ensemble_mean"),
            Vector{T}(undef, d),
        )
    centered_ensemble!(HA, measured_ens_mean, HX)
    return A, HX, HA
end

"""
    enkf_correct!(c, H, measurement_noise_dist, y, [v])

In-place correction step in an Ensemble Kalman filter (EnKF).

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `H::AbstractMatrix`: measurement matrix of the state space model
- `measurement_noise_dist::MvNormal`: Multivariate Gaussian measurement-noise distribution.
- `y::AbstractVector`: a measurement (data point)
- `v::AbstractVector` (optional): affine control input to the measurement

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter. Section 2.
"""
function enkf_correct!(
    c::FilteringCache,
    H::AbstractMatrix{T},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    d, D = size(H)
    N = get(c.entries, (typeof(D), size(D), "N")) do
        error("Ensemble size N is missing in FilteringCache.")
    end
    Nsub1 = N - 1
    forecast_ensemble = get(c.entries, (Matrix{T}, (D, N), "forecast_ensemble")) do
        error("Cannot correct, no forecast ensemble in cache.")
    end
    ensemble = get!(
        c.entries,
        (typeof(forecast_ensemble), size(forecast_ensemble), "ensemble"),
        similar(forecast_ensemble),
    )

    A, HX, HA = A_HX_HA!(c, H, v)
    # TODO: turn these into tests >>>>>>>>>>>>>>>>>>>>>>>>>v
    @assert A ≈ centered_ensemble(forecast_ensemble)
    @assert HX ≈ H * forecast_ensemble
    @assert HA ≈ H * A

    # D - HX = ([y + v_i]_i=1:N) - HX , with v_i ~ N(0, R)
    residual = get!(c.entries, (typeof(HX), size(HX), "residual"), similar(HX))
    Distributions.rand!(measurement_noise_dist, residual)
    residual .+= y
    residual .-= HX

    Σ̂ = get!(c.entries, (Matrix{T}, (D, D), "Σ̂"), Matrix{T}(undef, D, D))
    rdiv!(mul!(Σ̂, A, A'), Nsub1)
    cross_cov = get!(c.entries, (Matrix{T}, (D, d), "cross_cov"), Matrix{T}(undef, D, d))
    mul!(cross_cov, Symmetric(Σ̂), H')
    Ŝ = get!(c.entries, (Matrix{T}, (d, d), "Ŝ"), Matrix{T}(undef, d, d))
    mul!(Ŝ, H, cross_cov)
    Ŝ .+= measurement_noise_dist.Σ
    K̂ = cross_cov
    rdiv!(K̂, cholesky!(Symmetric(Ŝ, :L)))
    copy!(ensemble, forecast_ensemble)
    mul!(ensemble, K̂, residual, 1.0, 1.0)
    return ensemble
end

"""
    enkf_matrixfree_correct!(c, HX, HA, A, measurement_noise_dist, y; [R_inverse])

In-place observation-matrix-free correction step in an Ensemble Kalman filter (EnKF).

> Note:
> The arguments `A`, `HX`, `HA` can be computed using [`A_HX_HA!`](@ref)

# Arguments
- `c::FilteringCache`: Cache holding pre-allocated matrices and vectors
- `HX::AbstractMatrix`: measured forecast ensemble ``\\mathrm{H}\\mathrm{X} + v``
- `HA::AbstractMatrix`: measured and centered forecast ensemble ``\\mathrm{H}\\mathrm{A} + v``
- `A::AbstractMatrix`: centered forecast ensemble ``\\mathrm{A} = \\mathrm{X} - \\bar{\\mathrm{X}}``
- `measurement_noise_dist::MvNormal`: Multivariate Gaussian measurement-noise distribution.
- `y::AbstractVector`: a measurement (data point)
- `R_inverse::AbstractMatrix` (optional): the inverse ``R^{-1}`` of the measurement covariance matrix.
  `missing` per default.
   If `R_inverse` is not `missing`, then an alternative computation of the analysis ensemble is chosen, which uses
   the matrix-inversion lemma to scale as `O(N^3)`, instead of `O(d^3)`!

# References
[1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter. Sections 3, 4.
"""
function enkf_matrixfree_correct!(
    c::FilteringCache,
    HX::AbstractMatrix{T},
    HA::AbstractMatrix{T},
    A::AbstractMatrix{T},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{T};
    R_inverse::Union{AbstractMatrix{T},Missing} = missing,
) where {T}
    d = size(HX, 1)
    D, N = size(A)
    Nsub1 = N - 1
    forecast_ensemble = get(c.entries, (Matrix{T}, (D, N), "forecast_ensemble")) do
        error("Cannot correct, no forecast ensemble in cache.")
    end
    ensemble = get!(
        c.entries,
        (typeof(forecast_ensemble), size(forecast_ensemble), "ensemble"),
        similar(forecast_ensemble),
    )

    # D - HX = ([y + v_i]_i=1:N) - HX , with v_i ~ N(0, R)
    residual = get!(c.entries, (typeof(HX), size(HX), "residual"), similar(HX))
    Distributions.rand!(measurement_noise_dist, residual)
    residual .+= y
    residual .-= HX

    compute_P_inverse = !ismissing(R_inverse)
    if compute_P_inverse
        # ALLOCATE / QUERY CACHES >>>
        dxN_cache02 = get!(
            c.entries, (typeof(residual), size(residual), "dxN_001"),
            similar(residual),
        )
        dxN_cache03 = get!(
            c.entries, (typeof(residual), size(residual), "dxN_002"),
            similar(residual),
        )
        dxN_cache04 = get!(
            c.entries, (typeof(residual), size(residual), "dxN_003"),
            similar(residual),
        )
        NxN_cache01 = get!(
            c.entries, (Matrix{T}, (N, N), "NxN_000"), Matrix{T}(undef, N, N),
        )
        NxN_cache02 = get!(
            c.entries, (Matrix{T}, (N, N), "NxN_001"), Matrix{T}(undef, N, N),
        )
        # <<<

        # R⁻¹ (D - HX)
        mul!(dxN_cache02, R_inverse, residual)

        # (HA)' R⁻¹(D - HX)
        mul!(NxN_cache01, HA', dxN_cache02)

        # Q := I_N + (HA)'(R⁻¹/ (N-1)) (HA)
        #  -> R⁻¹(HA) / (N-1)
        rdiv!(mul!(dxN_cache03, R_inverse, HA), Nsub1)
        #  -> (HA)'R⁻¹(HA) / (N-1)
        mul!(NxN_cache02, HA', dxN_cache03) # That's Q without the added Identity matrix
        #  -> I_N + (HA)'R⁻¹(HA) / (N-1)
        @inbounds @simd for i in 1:N
            NxN_cache02[i, i] += 1.0
        end # So that's Q now

        # Q⁻¹ (HA)' R⁻¹(D - HX)   /GETS OVERWRITTEN\   /GETS OVERWRITTEN\
        #                        |by cholesky!| | by ldiv!  |
        ldiv!(cholesky!(Symmetric(NxN_cache02)), NxN_cache01)

        # K := (1/N-1) * (HA) Q⁻¹ (HA)' R⁻¹(D - HX)
        rdiv!(mul!(dxN_cache02, HA, NxN_cache01), Nsub1)

        # (D - HX) - K
        # copy!(dxN_cache01, residual)
        residual .-= dxN_cache02

        # R⁻¹((D - HX) - K) = P⁻¹(D - HX)
        mul!(dxN_cache04, R_inverse, residual)

        # (HA)' R⁻¹((D - HX) - K)
        mul!(NxN_cache02, HA', dxN_cache04)

        # Xᵃ = Xᶠ + A / (N-1) (HA)' R⁻¹((D - HX) - K)
        copy!(ensemble, forecast_ensemble)
        mul!(ensemble, rdiv!(A, Nsub1), NxN_cache02, 1.0, 1.0)
        return ensemble
    else
        # ALLOCATE / QUERY CACHES >>>
        P = get!(c.entries, (Matrix{T}, (d, d), "P"), Matrix{T}(undef, d, d))
        P_inv_times_res =
            get!(c.entries, (Matrix{T}, (d, N), "Pinv_times_res"), Matrix{T}(undef, d, N))
        NxN_cache = get!(c.entries, (Matrix{T}, (N, N), "NxN_000"), Matrix{T}(undef, N, N))
        # <<<
        ldiv!(Nsub1, mul!(P, HA, HA'))
        P .+= measurement_noise_dist.Σ
        ldiv!(P_inv_times_res, cholesky!(Symmetric(P)), residual)
        mul!(NxN_cache, HA', P_inv_times_res)
        copy!(ensemble, forecast_ensemble)
        mul!(ensemble, ldiv!(Nsub1, A), NxN_cache, 1.0, 1.0)
        return ensemble
    end
end

export enkf_predict!
export enkf_correct!
export enkf_matrixfree_correct!


function etkf_correct(
    forecast_ensemble::AbstractMatrix{tT},
    H::AbstractMatrix{tT},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{tT},
    v::Union{AbstractVector{tT},Missing} = missing,
    λ::tT = 1.0,
) where {tT}
    N = size(forecast_ensemble, 2)

    HX = H * forecast_ensemble
    if !ismissing(v)
        HX .+= v
    end

    R_chol = cholesky(measurement_noise_dist.Σ)
    HX_mean = ensemble_mean(HX)
    HX = HX .- HX_mean
    Zy = (1.0 / sqrt(N-1)) * HX
    Zy_T_Rinv_sqrt = (Zy' / R_chol.U)
    C, G, F_T = svd(Zy_T_Rinv_sqrt, full=true, alg=LinearAlgebra.QRIteration())
    G[G .< eps(tT)] .= 0.0
    if length(G) < N
        G = vcat(G, zeros(N - length(G)))
    end
    G = Diagonal(1.0 ./ sqrt.(1.0 .+ G.^2))
    T = C * G * C'

    forecast_mean = ensemble_mean(forecast_ensemble)
    Z_f = forecast_ensemble .- forecast_mean
    Z_a = (λ / sqrt(N - 1)) * Z_f * T

    K = Z_a * T' * Zy'

    whitened_residual = R_chol \ (y - HX_mean)
    analysis_mean = forecast_mean .+ K * whitened_residual
    analysis_ensemble = analysis_mean .+ sqrt(N - 1) * Z_a

    return analysis_ensemble
end

export etkf_correct


function denkf_correct(
    forecast_ensemble::AbstractMatrix{T},
    H::AbstractMatrix{T},
    measurement_noise_dist::MvNormal,
    y::AbstractVector{T},
    v::Union{AbstractVector{T},Missing} = missing,
) where {T}
    # Sakov and Oke, 2008. "A deterministic formulation of the ensemble Kalman filter...".
    # (i) Given the forecast ensemble Xf , calculate the ensemble mean,
    # or forecast xf by (4), and the ensemble anomalies Af by (5).
    forecast_mean = ensemble_mean(forecast_ensemble)
    Z_f = centered_ensemble(forecast_ensemble)
    # (ii) Calculate the analysis xa by using the Kalman analysis eq. (1).
    HX = H * forecast_ensemble
    # ---| Along the way: (iii) Calculate the analysed anomalies (for later)
    HZ_f = H * Z_f
    if !ismissing(v)
        HX .+= v
        HZ_f .+= v
    end
    HX_mean = ensemble_mean(HX)
    residual = y - HX_mean
    PH, HPH = _calc_PH_HPH(forecast_ensemble, H)
    Ŝ = HPH + measurement_noise_dist.Σ
    K̂ = PH / cholesky!(Symmetric(Ŝ))
    analysis_mean = forecast_mean + K̂ * residual
    Z_a = Z_f - 0.5 * K̂ * HZ_f
    # (iv) Calculate the analysed ensemble by offsetting the ana- lysed anomalies by the analysis:
    analysis_ensemble = analysis_mean .+ Z_a
    return analysis_ensemble
end

export denkf_correct


# function eakf_correct(
#     forecast_ensemble::AbstractMatrix{tT},
#     H::AbstractMatrix{tT},
#     measurement_noise_dist::MvNormal,
#     y::AbstractVector{tT},
#     v::Union{AbstractVector{tT},Missing} = missing,
#     λ::tT = 1.0,
# ) where {tT}
#     # Anderson, 2001. "An Ensemble Adjustment Kalman Filter for Data Assimilation".
#     # Appendix A.
#     N = size(forecast_ensemble, 2)
#     forecast_mean = ensemble_mean(forecast_ensemble)
#     Z_f = forecast_ensemble .- forecast_mean

#     HX = H * forecast_ensemble
#     if !ismissing(v)
#         HX .+= v
#     end

#     R_chol = cholesky(measurement_noise_dist.Σ)
#     HX_mean = ensemble_mean(HX)
#     HX = HX .- HX_mean
#     Zy = (1.0 / sqrt(N-1)) * HX
#     Zy_T_Rinv_sqrt = (Zy' / R_chol.U)
#     U, D, F_T = svd(Zy_T_Rinv_sqrt, full=true, alg=LinearAlgebra.QRIteration())
#     D[D .< eps(tT)] .= 0.0
#     if length(D) < N
#         D = vcat(D, zeros(N - length(D)))
#     end
#     B = Diagonal(1.0 ./ sqrt.(1.0 .+ D.^2))

#     F, G, F_T2 = svd((1.0 / sqrt(N-1)) * Z_f', full=true, alg=LinearAlgebra.QRIteration())

#     G[G .< eps(tT)] .= 0.0
#     if any(G .== 0.0)
#         error("Zero-singular values in Z_f decomposition.")
#     end
#     G_inv = 1.0 ./ G

#     if length(G) < N
#         G = vcat(G, zeros(N - length(G)))
#         G_inv = vcat(G_inv, zeros(N - length(G_inv)))
#     end

#     G = Diagonal(G)
#     G_inv = Diagonal(G_inv)

#     T = U * B * U'
#     A = F * G * T * G_inv * F'

#     Z_a = (λ / sqrt(N - 1)) * Z_f * A'

#     K = Z_a * T' * Zy'

#     whitened_residual = R_chol \ (y - HX_mean)
#     analysis_mean = forecast_mean .+ K * whitened_residual
#     analysis_ensemble = analysis_mean .+ sqrt(N - 1) * Z_a

#     return analysis_ensemble
# end

# export eakf_correct

