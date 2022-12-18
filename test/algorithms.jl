using Test
using LinearAlgebra
using Random
using Distributions
using ForwardDiff
using Plots

using PhDSE

const PLOT_RESULTS = false

"""
    gt:          size [T+1, D]
    obs:         size [T, d]
    estim_means: size [T+1, D]
"""
function plot_test(
    gt,
    obs,
    H,
    num_lines = missing;
    estim_means = missing,
    estim_stds = missing,
)
    dim_state = size(gt, 2)

    T_gt = 1:size(gt, 1)
    T_obs = 2:size(gt, 1)

    obs_idcs = H * collect(1:dim_state)
    s2m_idcs = indexin(1:dim_state, obs_idcs)
    if ismissing(num_lines)
        even_spacing = 1
    else
        @assert num_lines isa Int
        even_spacing = dim_state ÷ num_lines
    end

    gpl = plot()
    for l in 1:even_spacing:dim_state
        plot!(gpl, T_gt, gt[:, l], legend = false, color = "black")
        if !ismissing(estim_means)
            @assert size(estim_means) == size(gt)
            if !ismissing(estim_stds)
                @assert size(estim_stds) == size(gt)
                estim_ribbon = estim_stds[:, l]
            else
                estim_ribbon = nothing
            end
            plot!(
                gpl,
                T_gt,
                estim_means[:, l],
                ribbon = estim_ribbon,
                color = l,
                lw = 2,
                alpha = 0.5,
            )
        end
        if l ∈ obs_idcs
            scatter!(
                gpl,
                T_obs,
                obs[:, s2m_idcs[l]],
                markersize = 2,
                label = "",
                markershape = :x,
                color = l,
            )
        end
    end

    return gpl
end

function projectionmatrix(dimension::Int64, num_derivatives::Int64, derivative::Integer)
    kron(I(dimension), [i == (derivative + 1) ? 1.0 : 0.0 for i in 1:num_derivatives+1]')
end

function _matern(wiener_process_dimension, num_derivatives, lengthscale, dt)
    q = num_derivatives
    l = lengthscale

    ν = q - 1 / 2
    λ = sqrt(2ν) / l

    drift = diagm(1 => ones(q))
    @. drift[end, :] = -binomial(q + 1, 0:q) * λ^((q+1):-1:1)

    dispersion = zeros(q + 1)
    dispersion[end] = 1.0

    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exp(dt * M)
    A_breve = Mexp[1:d, 1:d]
    Q_breve = Mexp[1:d, d+1:end] * A_breve'

    A = kron(I(wiener_process_dimension), A_breve)
    Q = kron(I(wiener_process_dimension), Q_breve)
    @assert Q ≈ Q'
    sQ = Symmetric(0.5 * (Q + Q'))
    return A, sQ
end

function simulate_linear(
    A, Q, u, H, R, v, μ₀, Σ₀, N::Int,
)
    x = rand(Xoshiro(1), MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(Xoshiro(12), MvNormal(A * states[end] .+ u, Q)))
        push!(observations, rand(Xoshiro(34), MvNormal(H * states[end] .+ v, R)))
    end
    return states, observations
end

stack(x) = copy(reduce(hcat, x)')

function filtering_setup(D = 100, observe_every = 3, num_obs = 30)
    Random.seed!(1234)

    σ₀ = 0.001
    σᵣ = 0.03

    matern_derivs = 1
    totaldim = D * (matern_derivs + 1)

    μ₀ = rand(totaldim)
    Σ₀ = diagm(0 => σ₀ .* ones(totaldim))

    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / num_obs
    A, Q = _matern(D, matern_derivs, 1.0, dt)

    H = diagm(0 => ones(D))[1:observe_every:end, :] * projectionmatrix(D, matern_derivs, 0)
    measdim, totaldim = size(H)

    R = diagm(0 => σᵣ .* ones(measdim))

    u = 0.1 .* randn(totaldim)
    v = zeros(measdim)

    ground_truth, observations = simulate_linear(A, Q, u, H, R, v, μ₀, Σ₀, num_obs)

    if PLOT_RESULTS
        savefig(
            plot_test(stack(ground_truth), stack(observations), H),
            joinpath(mkpath("./out/"), "setup.png"),
        )
    end

    return μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations
end

@info "Executing tests for standard Kalman filters"
include("algorithms/kalman.jl")
@info "Executing tests for square-root Kalman filters"
include("algorithms/sqrt_kalman.jl")
@info "Executing tests for ensemble Kalman filters"
include("algorithms/enkf.jl")
