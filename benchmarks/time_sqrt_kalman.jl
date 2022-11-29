using Random
using LinearAlgebra
using Dates
using Profile
using BenchmarkTools

using PhDSE

D = 3000
d = 3000
N = 500

include("_setup.jl")
const Φ, Q, u, H, R, v, y, μ₀, Σ₀ = kalman_setup(D = D, d = d)

sqrt_Q = cholesky(Q).U
sqrt_R = cholesky(R).U

# Allocate memory
fcache = SqrtKFCache(D, d)
write_moments!(fcache; μ = μ₀, Σ = Σ₀)

# Output
_tstamp = Dates.format(now(), "yymmdd_HHMMSS")
directory_name = mkpath(joinpath(@__DIR__, "benchres_sqrt_kalman", _tstamp))
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
    println(io, "\n\nBenchmarks for correction step\n\n")
    show(io, MIME"text/plain"(), bres_correct)
end;

@info "Done."
