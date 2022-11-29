module PhDSE

using MKL

using Distributions
using LinearAlgebra

include("cache.jl")

# Algorithms
include("kalman.jl")
include("sqrt_kalman.jl")
include("ensemble_kalman.jl")

end
