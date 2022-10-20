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



## Example: Car-tracking

> Example adapted from the book "Bayesian Filtering and Smoothing" by Simo Särkkä.
### Build the state-space model...

<details>
<summary><b>Details ...</b></summary>

##### We define:
* The **dynamics** tracking position and velocities, each in x- & y-position.
* The **observation model** measures the position of the vehicle in x- & y-dimension
* The posterior is computed using a Kalman filter.

</details>

##### Build the state-space model

```julia
dt = 0.1
s1, s2 = 0.5, 0.5
q1, q2 = 1, 1

A = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1]

Q = [q1*dt^3/3 0 q1*dt^2/2 0;
     0 q2*dt^3/3 0 q2*dt^2/2;
     q1*dt^2/2 0 q1*dt 0;
     0 q2*dt^2/2 0 q2*dt]

H = [1 0 0 0; 0 1 0 0]

d, D = size(H)

R = [s1^2 0; 0 s2^2]

μ₀, Σ₀ = zeros(D), 2 * Matrix(1e-5 * I, D, D)
```

##### Generate data

> The ground-truth trajectory, as well as the noisy observations come from an auxiliary function `simulate`ing the dynamics. [Please have a look here](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/kalman_filter/), where you can also find **more examples**.

```julia
ground_truth, data = simulate(A, Q, zeros(D), H, R, zeros(d), μ₀, Σ₀, 200, rng=MersenneTwister(3))
```

### Allocate memory
In order to save time that would otherwise be necessary to allocate memory in the iterative algorithm, we do that once before the loop, and then **re-use** it.

> This is of course only really relevant for larger state spaces. Here, it's just used to teach the usage.

```julia
sol = [(μ₀, sqrt.(diag(Σ₀)))]
fcache = KFCache(D, d)
fcache.μ .= μ₀
fcache.Σ .= Σ₀
```

### Finally, run the algorithm ...

```julia
for y in data
    kf_predict!(fcache, A, Q)
    kf_correct!(fcache, H, R, y)
    push!(sol, (copy(fcache.μ), sqrt.(diag(fcache.Σ))))
end
```

### ... and plot the results

<details>
<summary><b>Show code</b></summary>

```julia
scatter([y[1] for y in data], [y[2] for y in data], label="Measurements", markersize=2)
plot!([y[1] for y in ground_truth], [y[2] for y in ground_truth], label="True Location", linewidth=4, alpha=0.8)
plot!(
    [y[1] for (y, s) in sol], [y[2] for (y, s) in sol],
    label="Filter Estimate",
    linewidth=4,
    alpha=0.8,
    legend=:bottomright,
)
```

</details>


![](https://github.com/schmidtjonathan/PhDSE.jl/blob/gh-pages/dev/examples/kalman_filter_example.svg)
