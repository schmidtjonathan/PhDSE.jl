# Ensemble Kalman filter for non-linear dynamics

```@example 1
using LinearAlgebra
using Random
using Distributions
using ForwardDiff
using Plots

using PhDSE
```

First, define a function that allows us to draw samples from the state-space model.

```@example 1
function simulate_nonlinear(
    f::Function,
    Q,
    h::Function,
    R,
    μ₀,
    Σ₀,
    N::Int;
    rng = Random.GLOBAL_RNG,
)
    x = rand(rng, MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(rng, MvNormal(f(states[end]), Q(x))))
        push!(observations, rand(rng, MvNormal(h(states[end]), R(x))))
    end
    return states, observations
end
nothing  # hide
```

Then, define the actual state-space model:

```@example 1
d, D = 1, 2
μ₀ = [-1.0, 1.0]
Σ₀ = [0.01 0.0
    0.0 0.01]
a, b, c = 0.2, 0.2, 3.0

function f(x)
    x1, x2 = x
    return [
        x1 + 0.1 * (c * (x1 - x1^3 / 3 + x2)),
        x2 + 0.1 * (-(1 / c) * (x1 - a - b * x2)),
    ]
end
function h(x)
    return x[1:1]
end

A(x) = ForwardDiff.jacobian(f, x)
Q(x) = Matrix{Float64}(0.001 * I(D))
H(x) = Matrix{Float64}(I(D))[1:1, :]
R(x) = Matrix{Float64}(I(d))
u(x) = f(x) - A(x) * x
v(x) = zeros(d)

nothing  # hide
```

Next, generate an example state trajectory and according measurements.

```@example 1
N = 200
ground_truth, observations = simulate_nonlinear(f, Q, h, R, μ₀, Σ₀, N)
nothing  # hide
```

Setup for the EnKF:

```@example 1
ENSEMBLE_SIZE = 200
init_dist = MvNormal(μ₀, Σ₀)
process_noise_dist(x) = MvNormal(zero(x), Q(x))
measurement_noise_dist(x) = MvNormal(zero(x), R(x))
nothing  # hide
```

Compute the filtering posterior...

```@example 1
ensemble = rand(init_dist, ENSEMBLE_SIZE)
enkf_cache = init_cache_ensemble!(FilteringCache(), ensemble)
enkf_traj = [(copy(μ₀), copy(Σ₀))]
for y in observations
    enkf_m, enkf_C = enkf_traj[end]
    enkf_ensemble = enkf_predict!(
        enkf_cache,
        A(enkf_m),
        process_noise_dist(enkf_m),
        u(enkf_m),
    )

    enkf_m, enkf_C = ensemble_mean_cov(enkf_ensemble)

    enkf_ensemble = enkf_correct!(
        enkf_cache,
        H(enkf_m),
        measurement_noise_dist(y),
        y,
        v(enkf_m),
    )

    enkf_m, enkf_C = ensemble_mean_cov(enkf_ensemble)

    push!(enkf_traj, (copy(enkf_m), copy(enkf_C)))
end
nothing  # hide
```

... and plot the results:
```@example 1
enkf_means = [m for (m, C) in enkf_traj]
enkf_stds = [2sqrt.(diag(C)) for (m, C) in enkf_traj]

plot_x1 = scatter(1:length(observations), [o[1] for o in observations], color = 1, label="data")
plot!(plot_x1, 1:length(ground_truth), [gt[1] for gt in ground_truth], label="gt", color=:black, lw=5, alpha=0.6)
plot_x2 = plot(1:length(ground_truth), [gt[2] for gt in ground_truth], label="gt", color=:black, lw=5, alpha=0.6)
plot!(
    plot_x1,
    1:length(enkf_means),
    [m[1] for m in enkf_means],
    ribbon = [s[1] for s in enkf_stds],
    label = "EnKF mean",
    color = 3,
    lw = 3,
)
plot!(
    plot_x2,
    1:length(enkf_means),
    [m[2] for m in enkf_means],
    ribbon = [s[2] for s in enkf_stds],
    label = "EnKF mean",
    color = 3,
    lw = 3,
)
res_plot = plot(plot_x1, plot_x2, layout = (1, 2))
savefig(res_plot, "ensemble_kalman_filter_example.svg")
nothing  # hide
```

![](ensemble_kalman_filter_example.svg)
