# PhDSE.jl

<div align="center">

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://schmidtjonathan.github.io/PhDSE.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://schmidtjonathan.github.io/PhDSE.jl/dev/)
[![Build Status](https://github.com/schmidtjonathan/PhDSE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/schmidtjonathan/PhDSE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/schmidtjonathan/PhDSE.jl/branch/main/graph/badge.svg?token=IIGAI706O1)](https://codecov.io/gh/schmidtjonathan/PhDSE.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![bench-shield](https://img.shields.io/badge/view-benchmarks-blueviolet)](./benchmarks/README.md)

</div>

**PhDSE** stands for **P**robabilistic **h**igh-**D**imensional **S**tate **E**stimation. Its aim is to provide runtime- and memory-efficient implementations of inference algorithms in probabilistic state-space models - all implemented in [Julia](https://julialang.org).

## Install

The package is in a very early phase of development and will likely be subject to significant changes in both the interface, as well as the inner workings.
Therefore, it is for now not registered as a Julia package, yet.

If you feel like playing around with the package a bit then you can install it directly from GitHub like this:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/schmidtjonathan/PhDSE.jl.git")
```

or

```julia-repl
] add https://github.com/schmidtjonathan/PhDSE.jl.git
```

## Algorithms

#### Bayesian Filtering
* [Kalman filter](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/kalman_filter/)
* [Square-root Kalman filter](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/sqrt_kalman_filter/)
* [Ensemble Kalman filter](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/ensemble_kalman_filter/)
* More to follow :raised_hands:



## Example with non-linear dynamics

### Build the state-space model...

<details>
<summary><b>Details ...</b></summary>

##### We define:
* The **dynamics** following the vector field of the FitzHugh-Nagumo equations
* The **observation model** measures the first component of the system
* The posterior is computed using a Kalman filter.

</details>

##### Build the state-space model

```julia
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
```

##### Generate data

> The ground-truth trajectory, as well as the noisy observations come from an auxiliary function `simulate`ing the dynamics. [Please have a look here](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/kalman_filter/), where you can also find **more examples**.

```julia
N = 200
ground_truth, observations = simulate_nonlinear(f, Q, h, R, μ₀, Σ₀, N)
```

### Allocate memory
In order to save time that would otherwise be necessary to allocate memory in the iterative algorithm, we do that once before the loop, and then **re-use** it.

> This is of course only really relevant for larger state spaces. Here, it's just used to teach the usage.

```julia
cache = init_cache_moments!(FilteringCache(), μ₀, Σ₀)
kf_traj = [(copy(μ₀), copy(Σ₀))]
```

### Finally, run the algorithm ...

```julia
for y in observations
    kf_m, kf_C = kf_traj[end]
    kf_m, kf_C = kf_predict!(
        cache,
        A(kf_m),
        Q(kf_m),
        u(kf_m),
    )

    kf_m, kf_C = kf_correct!(
        cache,
        H(kf_m),
        R(y),
        y,
        v(kf_m),
    )

    push!(kf_traj, (copy(kf_m), copy(kf_C)))
end
```

### ... and plot the results

<details>
<summary><b>Show code</b></summary>

```julia
kf_means = [m for (m, C) in kf_traj]
kf_stds = [2sqrt.(diag(C)) for (m, C) in kf_traj]

plot_x1 = scatter(1:length(observations), [o[1] for o in observations], color = 1, label="data")
plot!(plot_x1, 1:length(ground_truth), [gt[1] for gt in ground_truth], label="gt", color=:black, lw=5, alpha=0.6)
plot_x2 = plot(1:length(ground_truth), [gt[2] for gt in ground_truth], label="gt", color=:black, lw=5, alpha=0.6)
plot!(
    plot_x1,
    1:length(kf_means),
    [m[1] for m in kf_means],
    ribbon = [s[1] for s in kf_stds],
    label = "KF mean",
    color = 3,
    lw = 3,
)
plot!(
    plot_x2,
    1:length(kf_means),
    [m[2] for m in kf_means],
    ribbon = [s[2] for s in kf_stds],
    label = "KF mean",
    color = 3,
    lw = 3,
)
res_plot = plot(plot_x1, plot_x2, layout = (1, 2))
```

</details>


![](https://github.com/schmidtjonathan/PhDSE.jl/blob/gh-pages/dev/examples/kalman_filter_example.svg)
