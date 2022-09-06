using Random
using LinearAlgebra
using Distributions
using Dates
using Profile
using BenchmarkTools

using PhDSE

include("_setup.jl")

N = 1000
Rinv = inv(R)
# Allocate memory
fcache = EnKFCache(
    D,
    d,
    ensemble_size = N,
    process_noise_dist = MvNormal(zeros(D), Q),
    observation_noise_dist = MvNormal(zeros(d), R),
)
init_ensemble = rand(MvNormal(μ₀, Σ₀), N)
copy!(fcache.ensemble, init_ensemble)

# Output
_tstamp = Dates.format(now(), "yymmdd_HHMMSS")
directory_name = mkpath(joinpath(@__DIR__, "benchres_kalman", _tstamp))
@info "Saving output to" directory_name

# BENCHMARK ...
# ...prediction

@info "Benchmark predict step"
@info "--| precompile"
enkf_predict!(fcache, Φ, u)
@info "--| run benchmark"
Profile.clear()
bres_predict = @benchmark enkf_predict!($fcache, $Φ, $u)
show(stdout, MIME"text/plain"(), bres_predict)
println("\n")

# ...correction

@info "Benchmark correct step"
@info "--| precompile"
enkf_correct!(fcache, H, Rinv, y, v)
@info "--| run benchmark"
Profile.clear()
bres_correct = @benchmark enkf_correct!($fcache, $H, $Rinv, $y, $v)
show(stdout, MIME"text/plain"(), bres_correct)
println("\n")

open(joinpath(directory_name, "bres_out.txt"), "w") do io
    println(io, "Benchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_predict)
    println(io, "\n\nBenchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_correct)
end;

@info "Done."
