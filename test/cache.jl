using Test
using Distributions
using LinearAlgebra
using PhDSE

@testset "FilteringCache" begin
    d, D = 4, 8

    kf_cache = FilteringCache()
    @test kf_cache.entries isa IdDict

    _r = randn(D)
    m = randn(D)
    C = _r * _r' + 1e-7 * I(D)

    init_cache_moments!(kf_cache, m, C)
    @test haskey(kf_cache.entries, (typeof(m), size(m), "mean"))
    @test haskey(kf_cache.entries, (typeof(m), size(m), "predicted_mean"))
    @test haskey(kf_cache.entries, (typeof(C), size(C), "covariance"))
    @test haskey(kf_cache.entries, (typeof(C), size(C), "predicted_covariance"))

    @test kf_cache.entries[(typeof(m), size(m), "mean")] === kf_cache["mean"]
    @test kf_cache.entries[(typeof(m), size(m), "predicted_mean")] ===
          kf_cache["predicted_mean"]
    @test kf_cache.entries[(typeof(C), size(C), "covariance")] === kf_cache["covariance"]
    @test kf_cache.entries[(typeof(C), size(C), "predicted_covariance")] ===
          kf_cache["predicted_covariance"]

    fake_mean = randn(D + 1)
    kf_cache.entries[(typeof(fake_mean), size(fake_mean), "mean")] = fake_mean
    @test_throws ErrorException kf_cache["mean"]
    @test_throws KeyError kf_cache["non-existing-key"]

    # --

    ens_size = 100
    ens = rand(MvNormal(m, C), ens_size)
    enkf_cache = init_cache_ensemble!(FilteringCache(), ens)
    @test haskey(enkf_cache.entries, (typeof(ens_size), size(ens_size), "N"))
    @test haskey(enkf_cache.entries, (typeof(ens), size(ens), "ensemble"))
    @test haskey(enkf_cache.entries, (typeof(ens), size(ens), "forecast_ensemble"))

    # --
    structured_mat_cache = FilteringCache()
    U = cholesky(C).U
    init_cache_moments!(structured_mat_cache, copy(m), U)

    @test structured_mat_cache["covariance"] isa UpperTriangular
    @test structured_mat_cache["predicted_covariance"] isa UpperTriangular

    @test_throws ArgumentError mul!(structured_mat_cache["covariance"], randn(D, D), C)
end
