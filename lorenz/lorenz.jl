using LinearAlgebra
using Distributions
using Random
using ForwardDiff
using Plots
using OrdinaryDiffEq

using PhDSE

Random.seed!(20171027)

gr()

stack(x) = copy(reduce(hcat, x)')
const OUT_DIR = mkpath("./out/")

const ENSEMBLE_SIZE = 15

function plot_test(
    gt,
    obs,
    H,
    num_lines = 5;
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
    for (c, l) in enumerate(1:even_spacing:dim_state)
        @show l
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
                color = c,
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
                color = c,
            )
        end
    end

    return gpl
end

function filtering_setup_lorenz(dim_state=40)
    Random.seed!(142)

    function lorenz96(u, p, t)
        du = similar(u)
        N = length(u)
        F = p[1]
        # 3 edge cases explicitly (performance)
        @inbounds du[1] = (u[2] - u[N-1]) * u[N] - u[1] + F
        @inbounds du[2] = (u[3] - u[N]) * u[1] - u[2] + F
        @inbounds du[N] = (u[1] - u[N-2]) * u[N-1] - u[N] + F
        # then the general case
        for n in 3:(N-1)
            @inbounds du[n] = (u[n+1] - u[n-2]) * u[n-1] - u[n] + F
        end
        return du
    end

    dim_obs = dim_state #÷ 2
    initial_state_std = 1.0
    state_noise_std = 0.5
    observation_noise_std = 1.0
    force = 8.0
    θ = [force]

    tspan = (0.0, 5.0)
    num_data_points = 200

    Δt = (tspan[2] - tspan[1]) / num_data_points

    μ₀ = zeros(dim_state)
    u₀ = copy(μ₀)
    # u₀[1] += 0.01
    u₀ .+= 3 .* rand(size(u₀)...)
    Σ₀ = Matrix{Float64}(initial_state_std^2 * I(dim_state))
    function solve_lorenz_step(x, t)
        partial_prob = ODEProblem(lorenz96, x, (t, t + Δt), θ)
        fin = Array(solve(partial_prob, AutoTsit5(Rosenbrock23()); reltol=1e-10, abstol=1e-10, save_everystep=false, save_start = false))
        return vec(fin)
    end

    A(x, t) = ForwardDiff.jacobian(u -> solve_lorenz_step(u, t), x)
    Q = Matrix{Float64}(state_noise_std^2 * I(dim_state))
    u(x, t) = solve_lorenz_step(x, t) - A(x, t) * x
    H = Matrix{Float64}(I(dim_state))#[1:2:end, :]
    R = Matrix{Float64}(observation_noise_std^2 * I(dim_obs))
    v = zeros(dim_obs)

    # @assert size(H) == (dim_obs, dim_state)

    ground_truth = Array(
        solve(
            ODEProblem(lorenz96, u₀, tspan, θ),
            AutoTsit5(Rosenbrock23()),
            saveat = tspan[1]:Δt:tspan[2],
        ),
    )
    observations =
        H * ground_truth .+ rand(
            MvNormal(zeros(dim_obs), R), size(ground_truth, 2),
        )

    observations = observations'
    ground_truth = ground_truth'
    times = collect(tspan[1]:Δt:tspan[2])

    # @show size(observations) size(ground_truth) typeof(observations) typeof(ground_truth)
    @assert size(ground_truth, 2) == dim_state
    @assert size(observations, 2) == dim_obs
    @assert size(ground_truth, 1) == size(observations, 1) == length(times)

    return μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations, times
end


μ₀, Σ₀, A, Q, u, H, R, v, ground_truth, observations, times = filtering_setup_lorenz()
savefig(plot_test(ground_truth, observations, H), joinpath(OUT_DIR, "setup.png"))


function lorenz_kf()
    kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    for (t, y) in zip(times, [observations[i, :] for i in axes(observations, 1)])
        kf_m, kf_C = kf_predict(kf_m, kf_C, A(kf_m, t), Q, u(kf_m, t))
        kf_m, kf_C = kf_correct(kf_m, kf_C, H, R, y, v)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
    end

    kf_means = stack([m for (m, C) in kf_traj[2:end]])
    kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj[2:end]])

    res_plot = plot_test(
        ground_truth,
        observations,
        H;
        estim_means = kf_means,
        estim_stds = kf_stds,
    )

    savefig(res_plot, joinpath(OUT_DIR, "kf.png"))
    return res_plot
end

function lorenz_enkf()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(μ₀), Q)
    measurement_noise_dist = MvNormal(zero(observations[1, :]), R)
    ensemble = rand(init_dist, ENSEMBLE_SIZE)

    kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    for (t, y) in zip(times, [observations[i, :] for i in axes(observations, 1)])
        ensemble = enkf_predict(ensemble, A(kf_m, t), process_noise_dist, u(kf_m, t))

        kf_m, kf_C = ensemble_mean_cov(ensemble)

        ensemble = enkf_correct(ensemble, H, measurement_noise_dist, y, v)

        kf_m, kf_C = ensemble_mean_cov(ensemble)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
    end

    kf_means = stack([m for (m, C) in kf_traj[2:end]])
    kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj[2:end]])

    res_plot = plot_test(
        ground_truth,
        observations,
        H;
        estim_means = kf_means,
        estim_stds = kf_stds,
    )

    savefig(res_plot, joinpath(OUT_DIR, "enkf.png"))
    return res_plot
end


function lorenz_denkf()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(μ₀), Q)
    measurement_noise_dist = MvNormal(zero(observations[1, :]), R)
    ensemble = rand(init_dist, ENSEMBLE_SIZE)

    kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    for (t, y) in zip(times, [observations[i, :] for i in axes(observations, 1)])
        ensemble = enkf_predict(ensemble, A(kf_m, t), process_noise_dist, u(kf_m, t))

        kf_m, kf_C = ensemble_mean_cov(ensemble)

        ensemble = denkf_correct(ensemble, H, measurement_noise_dist, y, v)

        kf_m, kf_C = ensemble_mean_cov(ensemble)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
    end

    kf_means = stack([m for (m, C) in kf_traj[2:end]])
    kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj[2:end]])

    res_plot = plot_test(
        ground_truth,
        observations,
        H;
        estim_means = kf_means,
        estim_stds = kf_stds,
    )

    savefig(res_plot, joinpath(OUT_DIR, "denkf.png"))
    return res_plot
end


function lorenz_etkf()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(μ₀), Q)
    measurement_noise_dist = MvNormal(zero(observations[1, :]), R)
    ensemble = rand(init_dist, ENSEMBLE_SIZE)

    kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    for (t, y) in zip(times, [observations[i, :] for i in axes(observations, 1)])
        ensemble = enkf_predict(ensemble, A(kf_m, t), process_noise_dist, u(kf_m, t))

        kf_m, kf_C = ensemble_mean_cov(ensemble)

        ensemble = etkf_correct(ensemble, H, measurement_noise_dist, y, v)

        kf_m, kf_C = ensemble_mean_cov(ensemble)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
    end

    kf_means = stack([m for (m, C) in kf_traj[2:end]])
    kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj[2:end]])

    res_plot = plot_test(
        ground_truth,
        observations,
        H;
        estim_means = kf_means,
        estim_stds = kf_stds,
    )

    savefig(res_plot, joinpath(OUT_DIR, "etkf.png"))
    return res_plot
end

function lorenz_serial_etkf()
    init_dist = MvNormal(μ₀, Σ₀)
    process_noise_dist = MvNormal(zero(μ₀), Q)
    measurement_noise_dist = MvNormal(zero(observations[1, :]), R)
    ensemble = rand(init_dist, ENSEMBLE_SIZE)

    kf_m = copy(μ₀)
    kf_C = copy(Σ₀)
    kf_traj = [(copy(μ₀), copy(Σ₀))]
    for (t, y) in zip(times, [observations[i, :] for i in axes(observations, 1)])
        ensemble = enkf_predict(ensemble, A(kf_m, t), process_noise_dist, u(kf_m, t))

        kf_m, kf_C = ensemble_mean_cov(ensemble)

        ensemble =
            serial_etkf_correct(ensemble, H, measurement_noise_dist, y, v)

        kf_m, kf_C = ensemble_mean_cov(ensemble)
        push!(kf_traj, (copy(kf_m), copy(kf_C)))
    end

    kf_means = stack([m for (m, C) in kf_traj[2:end]])
    kf_stds = stack([2sqrt.(diag(C)) for (m, C) in kf_traj[2:end]])

    res_plot = plot_test(
        ground_truth,
        observations,
        H;
        estim_means = kf_means,
        estim_stds = kf_stds,
    )

    savefig(res_plot, joinpath(OUT_DIR, "serial_etkf.png"))
    return res_plot
end


@info "Kalman filter"
kf_plot = lorenz_kf()
@info "Ensemble Kalman filter with N = $ENSEMBLE_SIZE"
enkf_plot = lorenz_enkf()
@info "Deterministic Ensemble Kalman filter with N = $ENSEMBLE_SIZE"
denkf_plot = lorenz_denkf()
@info "Ensemble Transform Kalman filter with N = $ENSEMBLE_SIZE"
etkf_plot = lorenz_etkf()
@info "Serial Ensemble Transform Kalman filter with N = $ENSEMBLE_SIZE"
serial_etkf_plot = lorenz_serial_etkf()
# @info "Ensemble Adjustment Kalman filter with N = $ENSEMBLE_SIZE"
# eakf_plot = lorenz_eakf()
