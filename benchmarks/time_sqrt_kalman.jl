using Random
using LinearAlgebra
using PSDMatrices
using Dates
using Profile
using BenchmarkTools

using PhDSE

include("_setup.jl")
sqrt_Q = PSDMatrix(cholesky(Q).U)
sqrt_R = PSDMatrix(cholesky(R).U)
sqrt_Σ₀ = PSDMatrix(cholesky(Σ₀).U)

# Allocate memory
fcache = SqrtKFCache(D, d)
copy!(fcache.μ, μ₀)
copy!(fcache.Σ, sqrt_Σ₀)

# Output
_tstamp = Dates.format(now(), "yymmdd_HHMMSS")
directory_name = mkpath(joinpath(@__DIR__, "benchres_kalman", _tstamp))
@info "Saving output to" directory_name

# BENCHMARK ...
# ...prediction

@info "Benchmark sqrt-predict step"
@info "--| precompile"
sqrt_kf_predict!(fcache, Φ, sqrt_Q, u)
@info "--| run benchmark"
Profile.clear()
bres_predict = @benchmark sqrt_kf_predict!($fcache, $Φ, $sqrt_Q, $u)
show(stdout, MIME"text/plain"(), bres_predict)
println("\n")

# ...correction

@info "Benchmark sqrt-correct step"
@info "--| precompile"
sqrt_kf_correct!(fcache, H, sqrt_R, y, v)
@info "--| run benchmark"
Profile.clear()
bres_correct = @benchmark sqrt_kf_correct!($fcache, $H, $sqrt_R, $y, $v)
show(stdout, MIME"text/plain"(), bres_correct)
println("\n")

open(joinpath(directory_name, "bres_out.txt"), "w") do io
    println(io, "Benchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_predict)
    println(io, "\n\nBenchmarks for prediction step\n\n")
    show(io, MIME"text/plain"(), bres_correct)
end;

@info "Done."
