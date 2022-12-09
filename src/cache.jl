abstract type AbstractAlgCache end

RightMatrixSquareRoot = Union{Diagonal,UpperTriangular,UnitUpperTriangular,UniformScaling}
export RightMatrixSquareRoot

struct FilteringCache
    entries::IdDict{Tuple{Type,Tuple,AbstractString},Union{Number,AbstractArray}}
    FilteringCache() = new(IdDict{Tuple{Type,Tuple,AbstractString},Union{Number,AbstractArray}}())
end


function Base.getindex(c::FilteringCache, key::AbstractString)
    ret = missing
    ctr = 0
    for (k, v) in c.entries
        tp, sz, id_string = k
        if id_string == key
            if ctr > 0
                error("Found duplicate key in cache!")
            end
            ctr += 1
            ret = c.entries[k]
        end
    end
    if ismissing(ret)
        throw(KeyError(key))
    end
    return ret
end


function init_cache_moments!(c::FilteringCache, initial_mean::mT, initial_covariance::CT) where {mT <: AbstractArray, CT<:AbstractArray}
    # Set initial (filtering) moments
    setindex!(
        c.entries, copy(initial_mean), (typeof(initial_mean), size(initial_mean), "mean")
    )
    setindex!(
        c.entries, copy(initial_covariance), (typeof(initial_covariance), size(initial_covariance), "covariance")
    )
    # Allocate memory for predicted moments
    setindex!(
        c.entries, similar(initial_mean), (typeof(initial_mean), size(initial_mean), "predicted_mean")
    )
    setindex!(
        c.entries, similar(initial_covariance), (typeof(initial_covariance), size(initial_covariance), "predicted_covariance")
    )
    return c
end


function init_cache_ensemble!(c::FilteringCache, initial_ensemble::eT) where {eT <: AbstractArray}
    ensemble_size = size(initial_ensemble, 2)
    setindex!(
        c.entries, ensemble_size,  (typeof(ensemble_size), size(ensemble_size), "N")
    )
    # Set initial (filtering) ensemble
    setindex!(
        c.entries, copy(initial_ensemble), (typeof(initial_ensemble), size(initial_ensemble), "ensemble")
    )
    # Allocate memory for the forecast ensemble
    setindex!(
        c.entries, similar(initial_ensemble), (typeof(initial_ensemble), size(initial_ensemble), "forecast_ensemble")
    )
    return c
end


export FilteringCache
export init_cache_moments!
export init_cache_ensemble!
