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
