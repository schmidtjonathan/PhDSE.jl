using Random
using LinearAlgebra
using Dates
using Profile
using BenchmarkTools

using PhDSE

include("_setup.jl")

# Allocate memory
fcache = KFCache(D, d)
copy!(fcache.μ, μ₀)
copy!(fcache.Σ, Σ₀)

# Output
_tstamp = Dates.format(now(), "yymmdd_HHMMSS")
directory_name = mkpath(joinpath(@__DIR__, "benchres_kalman", _tstamp))
@info "Saving output to" directory_name

# BENCHMARK ...
# ...prediction

@info "Benchmark predict step"
@info "--| precompile"
kf_predict!(fcache, Φ, Q, u)
@info "--| run benchmark"
Profile.clear()
bres_predict = @benchmark kf_predict!($fcache, $Φ, $Q, $u)
show(stdout, MIME"text/plain"(), bres_predict)
println("\n")

# ...correction

@info "Benchmark correct step"
@info "--| precompile"
kf_correct!(fcache, H, R, y, v)
@info "--| run benchmark"
Profile.clear()
bres_correct = @benchmark kf_correct!($fcache, $H, $R, $y, $v)
show(stdout, MIME"text/plain"(), bres_correct)
println("\n")

open(joinpath(directory_name, "bres_out.txt"), "w") do io
    println(io, "Benchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_predict)
    println(io, "\n\nBenchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_correct)
end;

@info "Done."
