using PhDSE
using Test
using SafeTestsets

@testset "PhDSE.jl" begin
    @safetestset "Cache" begin
        include("cache.jl")
    end
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
end
