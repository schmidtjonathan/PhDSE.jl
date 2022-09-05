# PhDSE

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://schmidtjonathan.github.io/PhDSE.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://schmidtjonathan.github.io/PhDSE.jl/dev/)
[![Build Status](https://github.com/schmidtjonathan/PhDSE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/schmidtjonathan/PhDSE.jl/actions/workflows/CI.yml?query=branch%3Amain)


**PhDSE** stands for **P**robabilistic **h**igh-**D**imensional **S**tate **E**stimation. Its aim is to provide runtime- and memory-efficient implementations of inference algorithms in probabilistic state-space models - all implemented in [Julia](https://julialang.org).

# Install

The package is in a very early phase of development and will likely be subject to significant changes in both the interface, as well as the inner workings.
Therefore, it is for now not registered as a Julia package, yet.

If you feel like playing around with the package a bit then you can install it directly from GitHub like this:

```julia
using Pkg
Pkg.add(url="https://github.com/schmidtjonathan/PhDSE.jl.git")
```

or

```julia-repl
] add https://github.com/schmidtjonathan/PhDSE.jl.git
```

# Example

### Setting: Solving a partial differential equation (PDE)
We use a probabilistic numerical algorithm that is based on a Kalman filter to *solve a partial differential equation* (PDE).

<details>
<summary><b>Details</b></summary>

The equation is given as
$$\frac{\partial u(t, x)}{\partial t} = \nu \Delta u(t, x) =: F(t, x),$$
where $\Delta$ is the Laplace operator and $u(t, x)$ is the solution of our PDE.

We discretize the spatial independent variable $x$ on a finite grid $\mathbb{X}$ and use a finite-difference scheme to discretize $\Delta$. That leaves us with an ODE to solve $$\frac{\mathrm{d} u(t)}{\mathrm{d}t} = F(t, \mathbb{X}),$$
as described, e.g., [in this paper](https://proceedings.mlr.press/v151/kramer22a/kramer22a.pdf).

</details>

### Build the state-space model...

<details>
<summary><b>Details ...</b></summary>

...can be found in, e.g., [this paper](https://proceedings.mlr.press/v162/kramer22b/kramer22b.pdf).

##### (Very) brief summary:
* The **dynamics** come from a discretized integrated Brownian motion prior, which serves as a prior over the PDE solution and its first $q$ derivatives.
* The **observation model** measures the deviation between the modelled first derivative and the evaluation of the ODE vector field at the modelled ODE solution. If this deviation is zero (which we condition on (see `zero_data` below), then the model of the solution is a good candidate for the PDE solution.
* The posterior is computed using an (extended) Kalman filter.

</details>


```julia
dx = 0.01; x_grid = 0.0:dx:1.0  # spatial discretization
d = length(x_grid); q = 1  # dimensions, num. of derivatives
D = (q + 1) * d  # dimensionality of the state-space
t_0, t_max = (0.0, 0.3); dt = 1e-3  # temporal discretization

Φ, Q = discrete_Brownian_motion(d, q, dt)  # prior
proj0 = projectionmatrix(d, q, 0)  # | projection
proj1 = projectionmatrix(d, q, 1)  # | matrices

ν = 0.2
# measurement model & Jacobian
information_operator(u) = (proj1 * u) .- (ν .* Δ_1d(proj0 * u, dx))
information_operator_jac(u) = ForwardDiff.jacobian(information_operator, u)
R = 1e-10 * Matrix(I(d))

u0 =  exp.(-100 .* (x_grid .- 0.5).^2)  # | initial
u0_dot = ν .* Δ_1d(u0, dx)              # | conditions
U0 = intersperse([u0, u0_dot])          # | (solution & deriv.)

μ₀, Σ₀ = U0, 1e-10 .* Matrix(I(D))
zero_data = zeros(d)
```

> :warning: To see the implementation of the auxiliary functions used above, [please have a look here](https://schmidtjonathan.github.io/PhDSE.jl/dev/examples/solve_1d_heat_eq/), where you can also find **more examples**.

### Allocate memory
In order to save time that would otherwise be necessary to allocate memory in the iterative algorithm, we do that once before the loop, and then **re-use** it.

```julia
sol = [(t_0, copy(μ₀), copy(Σ₀))]
fcache = KFCache(D, d)
fcache.μ .= μ₀
fcache.Σ .= Σ₀
```

### Finally, run the algorithm ...

```julia
for (i, t) in enumerate(t_0:dt:t_max)
    # predict
    kf_predict!(fcache, Φ, Q)

    # linearize observations
    Hₜ = information_operator_jac(fcache.μ⁻)
    vₜ = information_operator(fcache.μ⁻) - Hₜ * fcache.μ⁻

    # measure
    kf_correct!(fcache, Hₜ, R, zero_data, vₜ)

    if i % 5 == 1
        push!(sol, (t, copy(fcache.μ), copy(fcache.Σ)))
    end
end
```

### ... and plot the results

<details>
<summary><b>Show code</b></summary>

```julia
anim = @animate for (t, μ, σ) in sol
	plot(
        x_grid,
        proj0 * μ,
        ylim=(-0.05, 1.0),
        linewidth=3,
        ribbon=1.97 .* stack([sqrt.(proj0 * diag(S)) for (t, m, S) in sol]),
        label="u(t)",
        title="t = $(round(t; digits=2))",
    )
end


gif(
	anim,
	"heat_eq_1d_example.gif",
	fps = 10,
)
```

</details>


![](https://github.com/schmidtjonathan/PhDSE.jl/blob/gh-pages/dev/examples/heat_eq_1d_example.gif)
