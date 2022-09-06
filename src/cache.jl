abstract type AbstractAlgCache end

Base.@kwdef struct KFCache{vT<:AbstractVector,mT<:AbstractMatrix} <: AbstractAlgCache
    #=
        D : state dimension
        d : measurement dimension
    =#

    # Predicted moments
    μ⁻::vT                      # D
    Σ⁻::mT                      # D x D
    # Corrected moments
    μ::vT                       # D
    Σ::mT                       # D x D

    # | Auxiliary
    # --| Prediction step
    # ----| Intermediate matmul result in prediction step
    predict_cache::mT           # D x D
    # --| Correction step
    # --| residual vector
    residual_cache::vT          # d
    # ----| Evaluation of the vector field
    obs_cache::vT                # d
    # ----| S matrix
    S_cache::mT                 # d x d
    # ----| Kalman gain K
    K_cache::mT               # D x d
    # ----| Intermediate matmul result in correction step
    correct_cache::mT           # d x D
end

function KFCache(state_dim::Int64, measurement_dim::Int64)
    return KFCache(
        μ⁻ = zeros(state_dim),
        Σ⁻ = zeros(state_dim, state_dim),
        μ = zeros(state_dim),
        Σ = zeros(state_dim, state_dim),
        predict_cache = zeros(state_dim, state_dim),
        residual_cache = zeros(measurement_dim),
        obs_cache = zeros(measurement_dim),
        S_cache = zeros(measurement_dim, measurement_dim),
        K_cache = zeros(state_dim, measurement_dim),
        correct_cache = zeros(state_dim, measurement_dim),
    )
end

export KFCache

# ==== SQRT Kalman filter

Base.@kwdef struct SqrtKFCache{vT<:AbstractVector,mT<:AbstractMatrix,psdT} <:
                   AbstractAlgCache
    #=
        D : state dimension
        d : measurement dimension
    =#

    # Predicted moments
    μ⁻::vT                      # D
    Σ⁻::psdT                      # D x D
    # Corrected moments
    μ::vT                       # D
    Σ::psdT                       # D x D

    # | Auxiliary
    cache_2DxD::mT
    cache_dpDxdpD::mT
    zero_cache_dxD::mT
    obs_cache::vT                # d
end

function SqrtKFCache(state_dim::Int64, measurement_dim::Int64)
    return SqrtKFCache(
        μ⁻ = zeros(state_dim),
        Σ⁻ = PSDMatrix(zeros(state_dim, state_dim)),
        μ = zeros(state_dim),
        Σ = PSDMatrix(zeros(state_dim, state_dim)),
        cache_2DxD = zeros(2state_dim, state_dim),
        cache_dpDxdpD = zeros(state_dim + measurement_dim, state_dim + measurement_dim),
        zero_cache_dxD = zeros(measurement_dim, state_dim),
        obs_cache = zeros(measurement_dim),
    )
end

export SqrtKFCache


# ==== Ensemble Kalman filter

Base.@kwdef struct EnKFCache{dT,mT<:AbstractMatrix} <: AbstractAlgCache
    #=
    D : state dimension
    d : measurement dimension
    N : ensemble size
    =#

    process_noise_dist::dT
    observation_noise_dist::dT

    ensemble::mT                # D x N
    forecast_ensemble::mT       # D x N
end

function EnKFCache(
    state_dim::Int64,
    measurement_dim::Int64;
    ensemble_size::Int64,
    process_noise_dist::Gaussian,
    observation_noise_dist::Gaussian
)
    return EnKFCache(
        process_noise_dist = process_noise_dist,
        observation_noise_dist = observation_noise_dist,
        ensemble = zeros(state_dim, ensemble_size),
        forecast_ensemble = zeros(state_dim, ensemble_size),
    )
end

export EnKFCache
