using PhDSE
using Test
using SafeTestsets

@testset "PhDSE.jl" begin
    @safetestset "Algorithms" begin
        include("algorithms.jl")
    end
end
