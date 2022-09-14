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

abstract type AbstractEnKFCache <: AbstractAlgCache end
export AbstractEnKFCache

Base.@kwdef struct EnKFCache{
    dT1<:MultivariateDistribution,
    dT2<:MultivariateDistribution,
    mT<:AbstractMatrix,
    vT<:AbstractVector,
} <: AbstractEnKFCache
    #=
    D : state dimension
    d : measurement dimension
    N : ensemble size
    =#

    process_noise_dist::dT1
    observation_noise_dist::dT2

    ensemble::mT                # D x N
    forecast_ensemble::mT       # D x N

    perturbed_D::mT             # d x N
    A::mT                       # D x N
    HX::mT                      # d x N
    HA::mT                      # d x N
    mX::vT                      # D

    Q::mT                       # N x N
    W::mT                       # N x d
    M::mT                       # d x d
    HAt_x_Rinv::mT              # N x d
    HAt_x_S_inv::mT             # N x d
    AHAt_x_Sinv::mT             # D x d
    residual::mT                # d x N
end

function EnKFCache(
    state_dim::Int64,
    measurement_dim::Int64;
    ensemble_size::Int64,
    process_noise_dist::MvNormal,
    observation_noise_dist::MvNormal,
)
    return EnKFCache(
        process_noise_dist = process_noise_dist,
        observation_noise_dist = observation_noise_dist,
        ensemble = zeros(state_dim, ensemble_size),
        forecast_ensemble = zeros(state_dim, ensemble_size),
        perturbed_D = zeros(measurement_dim, ensemble_size),
        A = zeros(state_dim, ensemble_size),
        HX = zeros(measurement_dim, ensemble_size),
        HA = zeros(measurement_dim, ensemble_size),
        mX = zeros(state_dim),
        Q = zeros(ensemble_size, ensemble_size),
        W = zeros(ensemble_size, measurement_dim),
        M = zeros(measurement_dim, measurement_dim),
        HAt_x_Rinv = zeros(ensemble_size, measurement_dim),
        HAt_x_S_inv = zeros(ensemble_size, measurement_dim),
        AHAt_x_Sinv = zeros(state_dim, measurement_dim),
        residual = zeros(measurement_dim, ensemble_size),
    )
end

export EnKFCache

Base.@kwdef struct OMFEnKFCache{
    dT1<:MultivariateDistribution,
    dT2<:MultivariateDistribution,
    mT<:AbstractMatrix,
    vT<:AbstractVector,
} <: AbstractEnKFCache
    #=
    D : state dimension
    d : measurement dimension
    N : ensemble size
    =#

    process_noise_dist::dT1
    observation_noise_dist::dT2

    ensemble::mT                 # D x N
    forecast_ensemble::mT        # D x N
    mX::vT                       # D
    A::mT                        # D x N

    dxN_cache01::mT              # d x N
    dxN_cache02::mT              # d x N
    dxN_cache03::mT              # d x N
    dxN_cache04::mT              # d x N
    NxN_cache01::mT              # N x N
    NxN_cache02::mT              # N x N
    DxN_cache01::mT              # D x N
end

function OMFEnKFCache(
    state_dim::Int64,
    measurement_dim::Int64;
    ensemble_size::Int64,
    process_noise_dist::MvNormal,
    observation_noise_dist::MvNormal,
)
    return OMFEnKFCache(
        process_noise_dist = process_noise_dist,
        observation_noise_dist = observation_noise_dist,
        ensemble = zeros(state_dim, ensemble_size),
        forecast_ensemble = zeros(state_dim, ensemble_size),
        mX = zeros(state_dim),
        A = zeros(state_dim, ensemble_size),
        dxN_cache01 = zeros(measurement_dim, ensemble_size),
        dxN_cache02 = zeros(measurement_dim, ensemble_size),
        dxN_cache03 = zeros(measurement_dim, ensemble_size),
        dxN_cache04 = zeros(measurement_dim, ensemble_size),
        NxN_cache01 = zeros(ensemble_size, ensemble_size),
        NxN_cache02 = zeros(ensemble_size, ensemble_size),
        DxN_cache01 = zeros(state_dim, ensemble_size),
    )
end

export OMFEnKFCache
