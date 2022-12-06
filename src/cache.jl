abstract type AbstractAlgCache end

RightMatrixSquareRoot = Union{Diagonal,UpperTriangular,UnitUpperTriangular,UniformScaling}
export RightMatrixSquareRoot

struct FilteringCache
    entries::IdDict{Tuple{Type,Tuple,String},AbstractArray}
end

function FilteringCache(; initial_mean::AbstractVector, initial_covariance::AbstractMatrix)
    @info "Building KF Cache"
    FilteringCache(
        IdDict{Tuple{Type,Tuple,AbstractString},AbstractArray}(
            (typeof(initial_mean), size(initial_mean), "mean") => copy(initial_mean),
            (typeof(initial_mean), size(initial_mean), "predicted_mean") =>
                similar(initial_mean),
            (typeof(initial_covariance), size(initial_covariance), "covariance") =>
                copy(initial_covariance),
            (
                typeof(initial_covariance),
                size(initial_covariance),
                "predicted_covariance",
            ) => similar(initial_covariance),
        ),
    )
end

function FilteringCache(; ensemble::AbstractMatrix)
    @info "Building EnKF Cache"
    FilteringCache(
        IdDict{Tuple{Type,Tuple,AbstractString},AbstractArray}(
            (typeof(ensemble), size(ensemble), "forecast_ensemble") => similar(ensemble),
            (typeof(ensemble), size(ensemble), "ensemble") => copy(ensemble),
        ),
    )
end

export FilteringCache

# ==== Ensemble Kalman filter

Base.@kwdef struct EnKFCache{
    dT1<:MultivariateDistribution,
    dT2<:MultivariateDistribution,
    mT<:AbstractMatrix,
    vT<:AbstractVector,
} <: AbstractAlgCache
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
    HX::mT                       # d x N
    HA::mT                       # d x N

    dxN_cache01::mT              # d x N
    dxN_cache02::mT              # d x N
    dxN_cache03::mT              # d x N
    dxN_cache04::mT              # d x N
    NxN_cache01::mT              # N x N
    NxN_cache02::mT              # N x N
    DxN_cache01::mT              # D x N
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
        mX = zeros(state_dim),
        A = zeros(state_dim, ensemble_size),
        HX = zeros(measurement_dim, ensemble_size),
        HA = zeros(measurement_dim, ensemble_size),
        dxN_cache01 = zeros(measurement_dim, ensemble_size),
        dxN_cache02 = zeros(measurement_dim, ensemble_size),
        dxN_cache03 = zeros(measurement_dim, ensemble_size),
        dxN_cache04 = zeros(measurement_dim, ensemble_size),
        NxN_cache01 = zeros(ensemble_size, ensemble_size),
        NxN_cache02 = zeros(ensemble_size, ensemble_size),
        DxN_cache01 = zeros(state_dim, ensemble_size),
    )
end

export EnKFCache

# function write_moments!(cache::SqrtKFCache; μ = missing, Σ = missing)
#     if !ismissing(μ)
#         size(μ) == size(cache.μ) || error(
#             "Cannot write mean of size $(size(μ)) to cache entry of size $(size(cache.μ)).",
#         )
#         copy!(cache.μ, μ)
#     end
#     if !ismissing(Σ)
#         size(Σ) == size(cache.Σ) || error(
#             "Cannot write cov of size $(size(Σ)) to cache entry of size $(size(cache.Σ)).",
#         )
#         if Σ isa RightMatrixSquareRoot
#             copy!(cache.Σ, Σ)
#         elseif Σ isa Matrix
#             U = cholesky(Σ).U
#             copy!(cache.Σ, U)
#         else
#             error("Cannot set covariance of type $(typeof(Σ)) in SqrtKFCache.")
#         end
#     end
#     return cache
# end

function write_moments!(cache::EnKFCache; μ = missing, Σ = missing)
    if ismissing(μ) || ismissing(Σ)
        error("Need both μ and Σ to be provided to set ensemble in EnKFCache.")
    end
    size(μ) == (size(cache.ensemble, 1),) || error(
        "Cannot write mean of size $(size(μ)) to cache entry of size $(size(cache.ensemble)).",
    )
    size(Σ) == (size(cache.ensemble, 1), size(cache.ensemble, 1)) || error(
        "Cannot write cov of size $(size(Σ)) to cache entry of size $(size(cache.ensemble)).",
    )
    N = size(cache.ensemble, 2)
    ens = rand(MvNormal(μ, Σ), N)
    copy!(cache.ensemble, ens)
    return cache
end

export write_moments!
