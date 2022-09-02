# Advanced example: Solve a partial differential equation

```@example 1
using LinearAlgebra
using Random
using GaussianDistributions
using ForwardDiff
using ToeplitzMatrices

using Plots

using PhDSE
```

We first define some utility functions.

```@example 1
intersperse(vs) = reduce(hcat, vs)'[:]
stack(x) = copy(reduce(hcat, x)')

"""
Computes the system matrices for a discretized q-times-integrated Brownian-motion prior over our dynamics.

Credits: Nathanael Bosch (https://github.com/nathanaelbosch/KalmanFilterToolbox.jl)
"""
function discrete_Brownian_motion(dimension::Int64, num_derivatives::Int64, dt::Real)
    v = 0:num_derivatives
    f = factorial.(v)
    A_breve = TriangularToeplitz(dt .^ v ./ f, :U) |> Matrix
    e = (2 * num_derivatives + 1 .- v .- v')
    fr = reverse(f)
    Q_breve = @. dt^e / (e * fr * fr')

    A = kron(I(dimension), A_breve)
    Q = kron(I(dimension), Q_breve)
    return A, Q

end

"""
Computes projection matrices from the state vector to the respective derivatives.
"""
function projectionmatrix(dimension::Int64, num_derivatives::Int64, derivative::Integer)
    kron(diagm(0 => ones(dimension)), [i == (derivative + 1) ? 1 : 0 for i in 1:num_derivatives+1]')
end

"""
Compute the finite-differences-discretized Laplace operator in one dimension.
"""
function Δ_1d(u, δx; bound_val=0.0)
	Δu = copy(u)
	@simd for i in 2:length(u)-1
		Δu[i] = (u[i+1] - 2u[i] + u[i-1]) / (δx^2)
	end
	return Δu
end
nothing # hide
```

Set up the state space model

```@example 1
dx = 0.01
x_grid = 0.0:dx:1.0

d = length(x_grid)
q = 1
D = (q + 1) * d

t_0, t_max = (0.0, 0.3)
dt = 1e-3

Φ, Q = discrete_Brownian_motion(d, q, dt)

proj0 = projectionmatrix(d, q, 0)
proj1 = projectionmatrix(d, q, 1)

ν = 0.2
information_operator(u) = (proj1 * u) .- (ν .* Δ_1d(proj0 * u, dx))
information_operator_jac(u) = ForwardDiff.jacobian(information_operator, u)

R = 1e-10 * Matrix(I(d))


u0 =  exp.(-100 .* (x_grid .- 0.5).^2) # .* (1.0./(1.0 .+ exp.(-100 .* (x_grid .- 0.45))))
u0_dot = ν .* Δ_1d(u0, dx)
U0 = intersperse([u0, u0_dot])

μ₀, Σ₀ = U0, 1e-10 .* Matrix(I(D))
zero_data = zeros(d)
nothing # hide
```

Initialize the cache ...

```@example 1
sol = [(t_0, copy(μ₀), copy(Σ₀))]
fcache = KFCache(D, d)
fcache.μ .= μ₀
fcache.Σ .= Σ₀
nothing # hide
```

... and start filtering!

```@example 1
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

Finally, plot the results!

```@example 1
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
