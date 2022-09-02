var documenterSearchIndex = {"docs":
[{"location":"algorithms/sqrt_kalman_filter/#Square-root-Kalman-Filter","page":"Square-root Kalman filter","title":"Square-root Kalman Filter","text":"","category":"section"},{"location":"algorithms/sqrt_kalman_filter/","page":"Square-root Kalman filter","title":"Square-root Kalman filter","text":"sqrt_kf_predict!\nsqrt_kf_correct!","category":"page"},{"location":"algorithms/sqrt_kalman_filter/#PhDSE.sqrt_kf_predict!","page":"Square-root Kalman filter","title":"PhDSE.sqrt_kf_predict!","text":"sqrt_kf_predict!(fcache, Φ, Q, [u])\n\nEfficient and in-place implementation of the prediction step in a square-root Kalman filter.\n\nArguments\n\nfcache::KFCache: a cache holding memory-heavy objects\nΦ::AbstractMatrix: transition matrix, i.e. dynamics of the state space model\nQ::PSDMatrix: right matrix square root of transition covariance, i.e. process noise of the state space model\nu::AbstractVector (optional): affine control input to the dynamics\n\n\n\n\n\n","category":"function"},{"location":"algorithms/sqrt_kalman_filter/#PhDSE.sqrt_kf_correct!","page":"Square-root Kalman filter","title":"PhDSE.sqrt_kf_correct!","text":"sqrt_kf_correct!(fcache, H, R, y, [v])\n\nEfficient and in-place implementation of the correction step in a square-root Kalman filter.\n\nArguments\n\nfcache::KFCache: a cache holding memory-heavy objects\ny::AbstractVector: a measurement (data point)\nH::AbstractMatrix: measurement matrix of the state space model\nR::PSDMatrix: right matrix square root of measurement noise covariance of the state space model\nv::AbstractVector (optional): affine control input to the measurement\n\n\n\n\n\n","category":"function"},{"location":"algorithms/kalman_filter/#Kalman-Filter","page":"Kalman Filter","title":"Kalman Filter","text":"","category":"section"},{"location":"algorithms/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"kf_predict!\nkf_correct!","category":"page"},{"location":"algorithms/kalman_filter/#PhDSE.kf_predict!","page":"Kalman Filter","title":"PhDSE.kf_predict!","text":"kf_predict!(fcache, Φ, [u], Q)\n\nEfficient and in-place implementation of the prediction step in a Kalman filter.\n\nArguments\n\nfcache::KFCache: a cache holding memory-heavy objects\nΦ::AbstractMatrix: transition matrix, i.e. dynamics of the state space model\nu::AbstractVector (optional): affine control input to the dynamics\nQ::AbstractMatrix: transition covariance, i.e. process noise of the state space model\n\n\n\n\n\n","category":"function"},{"location":"algorithms/kalman_filter/#PhDSE.kf_correct!","page":"Kalman Filter","title":"PhDSE.kf_correct!","text":"kf_correct!(fcache, y, H, [v], R)\n\nEfficient and in-place implementation of the correction step in a Kalman filter.\n\nArguments\n\nfcache::KFCache: a cache holding memory-heavy objects\ny::AbstractVector: a measurement (data point)\nH::AbstractMatrix: measurement matrix of the state space model\nv::AbstractVector (optional): affine control input to the measurement\nR::AbstractMatrix: measurement noise covariance of the state space model\n\n\n\n\n\n","category":"function"},{"location":"examples/solve_1d_heat_eq/#Advanced-example:-Solve-a-partial-differential-equation","page":"Advanced: Solve PDE","title":"Advanced example: Solve a partial differential equation","text":"","category":"section"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"using LinearAlgebra\nusing Random\nusing GaussianDistributions\nusing ForwardDiff\nusing ToeplitzMatrices\n\nusing Plots\n\nusing PhDSE","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"We first define some utility functions.","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"intersperse(vs) = reduce(hcat, vs)'[:]\nstack(x) = copy(reduce(hcat, x)')\n\n\"\"\"\nComputes the system matrices for a discretized q-times-integrated Brownian-motion prior over our dynamics.\n\nCredits: Nathanael Bosch (https://github.com/nathanaelbosch/KalmanFilterToolbox.jl)\n\"\"\"\nfunction discrete_Brownian_motion(dimension::Int64, num_derivatives::Int64, dt::Real)\n    v = 0:num_derivatives\n    f = factorial.(v)\n    A_breve = TriangularToeplitz(dt .^ v ./ f, :U) |> Matrix\n    e = (2 * num_derivatives + 1 .- v .- v')\n    fr = reverse(f)\n    Q_breve = @. dt^e / (e * fr * fr')\n\n    A = kron(I(dimension), A_breve)\n    Q = kron(I(dimension), Q_breve)\n    return A, Q\n\nend\n\n\"\"\"\nComputes projection matrices from the state vector to the respective derivatives.\n\"\"\"\nfunction projectionmatrix(dimension::Int64, num_derivatives::Int64, derivative::Integer)\n    kron(diagm(0 => ones(dimension)), [i == (derivative + 1) ? 1 : 0 for i in 1:num_derivatives+1]')\nend\n\n\"\"\"\nCompute the finite-differences-discretized Laplace operator in one dimension.\n\"\"\"\nfunction Δ_1d(u, δx; bound_val=0.0)\n\tΔu = copy(u)\n\t@simd for i in 2:length(u)-1\n\t\tΔu[i] = (u[i+1] - 2u[i] + u[i-1]) / (δx^2)\n\tend\n\treturn Δu\nend\nnothing # hide","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"Set up the state space model","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"dx = 0.01\nx_grid = 0.0:dx:1.0\n\nd = length(x_grid)\nq = 1\nD = (q + 1) * d\n\nt_0, t_max = (0.0, 0.3)\ndt = 1e-3\n\nΦ, Q = discrete_Brownian_motion(d, q, dt)\n\nproj0 = projectionmatrix(d, q, 0)\nproj1 = projectionmatrix(d, q, 1)\n\nν = 0.2\ninformation_operator(u) = (proj1 * u) .- (ν .* Δ_1d(proj0 * u, dx))\ninformation_operator_jac(u) = ForwardDiff.jacobian(information_operator, u)\n\nR = 1e-10 * Matrix(I(d))\n\n\nu0 =  exp.(-100 .* (x_grid .- 0.5).^2) # .* (1.0./(1.0 .+ exp.(-100 .* (x_grid .- 0.45))))\nu0_dot = ν .* Δ_1d(u0, dx)\nU0 = intersperse([u0, u0_dot])\n\nμ₀, Σ₀ = U0, 1e-10 .* Matrix(I(D))\nzero_data = zeros(d)\nnothing # hide","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"Initialize the cache ...","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"sol = [(t_0, copy(μ₀), copy(Σ₀))]\nfcache = KFCache(D, d)\nfcache.μ .= μ₀\nfcache.Σ .= Σ₀\nnothing # hide","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"... and start filtering!","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"for (i, t) in enumerate(t_0:dt:t_max)\n    # predict\n    kf_predict!(fcache, Φ, Q)\n\n    # linearize observations\n    Hₜ = information_operator_jac(fcache.μ⁻)\n    vₜ = information_operator(fcache.μ⁻) - Hₜ * fcache.μ⁻\n\n    # measure\n    kf_correct!(fcache, Hₜ, R, zero_data, vₜ)\n\n    if i % 5 == 1\n        push!(sol, (t, copy(fcache.μ), copy(fcache.Σ)))\n    end\nend","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"Finally, plot the results!","category":"page"},{"location":"examples/solve_1d_heat_eq/","page":"Advanced: Solve PDE","title":"Advanced: Solve PDE","text":"anim = @animate for (t, μ, σ) in sol\n\tplot(\n        x_grid,\n        proj0 * μ,\n        ylim=(-0.05, 1.0),\n        linewidth=3,\n        ribbon=1.97 .* stack([sqrt.(proj0 * diag(S)) for (t, m, S) in sol]),\n        label=\"u(t)\",\n        title=\"t = $(round(t; digits=2))\",\n    )\nend\n\n\ngif(\n\tanim,\n\t\"heat_eq_1d_example.gif\",\n\tfps = 10,\n)","category":"page"},{"location":"#PhDSE","page":"Home","title":"PhDSE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Probabilistic high-Dimensional State Estimation","category":"page"},{"location":"examples/kalman_filter/#Kalman-filter-for-car-tracking","page":"Kalman Filter","title":"Kalman filter for car tracking","text":"","category":"section"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"From \"Bayesian Filtering and Smoothing\" [1], example 4.3.","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"using LinearAlgebra\nusing Random\nusing GaussianDistributions\n\nusing Plots\n\nusing PhDSE","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"First, set up the state space model.","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"function simulate(Φ, Q, u, H, R, v, μ₀, Σ₀, N; rng = Random.GLOBAL_RNG)\n    x = rand(rng, Gaussian(μ₀, Σ₀))\n    states = [x]\n    observations = []\n\n    for i in 1:N\n        push!(states, rand(rng, Gaussian(Φ * states[end] + u, Q)))\n        push!(observations, rand(rng, Gaussian(H * states[end] + v, R)))\n    end\n    return states, observations\nend\n\n\ndt = 0.1\ns1, s2 = 0.5, 0.5\nq1, q2 = 1, 1\n\nA = [1 0 dt 0;\n     0 1 0 dt;\n     0 0 1 0;\n     0 0 0 1]\n\nQ = [q1*dt^3/3 0 q1*dt^2/2 0;\n     0 q2*dt^3/3 0 q2*dt^2/2;\n     q1*dt^2/2 0 q1*dt 0;\n     0 q2*dt^2/2 0 q2*dt]\n\nH = [1 0 0 0; 0 1 0 0]\n\nd, D = size(H)\n\nR = [s1^2 0; 0 s2^2]\n\nμ₀, Σ₀ = zeros(D), 2 * Matrix(1e-5 * I, D, D)\nnothing # hide","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"Next, generate an example state trajectory and according measurements.","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"ground_truth, data = simulate(A, Q, zeros(D), H, R, zeros(d), μ₀, Σ₀, 200, rng=MersenneTwister(3))\nnothing # hide","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"Compute the filtering posterior.","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"sol = [(μ₀, sqrt.(diag(Σ₀)))]\nfcache = KFCache(D, d)\nfcache.μ .= μ₀\nfcache.Σ .= Σ₀\nfor y in data\n    kf_predict!(fcache, A, Q)\n    kf_correct!(fcache, H, R, y)\n    push!(sol, (copy(fcache.μ), sqrt.(diag(fcache.Σ))))\nend\nnothing # hide","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"Finally, plot the results.","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"scatter([y[1] for y in data], [y[2] for y in data], label=\"Measurements\", markersize=2)\nplot!([y[1] for y in ground_truth], [y[2] for y in ground_truth], label=\"True Location\", linewidth=4, alpha=0.8)\nplot!(\n    [y[1] for (y, s) in sol], [y[2] for (y, s) in sol],\n    ribbon=(1.96 .* [s[1] for (y, s) in sol], 1.96 .* [s[2] for (y, s) in sol]),\n    label=\"Filter Estimate\", linewidth=4, alpha=0.8\n)\nsavefig(\"kalman_filter_example.svg\")\nnothing # hide","category":"page"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"(Image: )","category":"page"},{"location":"examples/kalman_filter/#References","page":"Kalman Filter","title":"References","text":"","category":"section"},{"location":"examples/kalman_filter/","page":"Kalman Filter","title":"Kalman Filter","text":"[1] \"Bayesian Filtering and Smoothing\", Simo Särkka, Cambridge University Press, 2013.","category":"page"},{"location":"examples/sqrt_kalman_filter/#Square-root-Kalman-filter-for-car-tracking","page":"Square-root Kalman Filter","title":"Square-root Kalman filter for car tracking","text":"","category":"section"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"From \"Bayesian Filtering and Smoothing\" [1], example 4.3.","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"using LinearAlgebra\nusing Random\nusing GaussianDistributions\nusing PSDMatrices\n\nusing Plots\n\nusing PhDSE","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"First, set up the state space model.","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"function simulate(Φ, Q, u, H, R, v, μ₀, Σ₀, N; rng = Random.GLOBAL_RNG)\n    x = rand(rng, Gaussian(μ₀, Σ₀))\n    states = [x]\n    observations = []\n\n    for i in 1:N\n        push!(states, rand(rng, Gaussian(Φ * states[end] + u, Q)))\n        push!(observations, rand(rng, Gaussian(H * states[end] + v, R)))\n    end\n    return states, observations\nend\n\n\ndt = 0.1\ns1, s2 = 0.5, 0.5\nq1, q2 = 1, 1\n\nA = [1 0 dt 0;\n     0 1 0 dt;\n     0 0 1 0;\n     0 0 0 1]\n\nQ = [q1*dt^3/3 0 q1*dt^2/2 0;\n     0 q2*dt^3/3 0 q2*dt^2/2;\n     q1*dt^2/2 0 q1*dt 0;\n     0 q2*dt^2/2 0 q2*dt]\n\nsqrt_Q = PSDMatrix(cholesky(Q).U)\n\nH = [1 0 0 0; 0 1 0 0]\n\nd, D = size(H)\n\nR = [s1^2 0; 0 s2^2]\nsqrt_R = PSDMatrix(cholesky(R).U)\n\nμ₀, Σ₀ = zeros(D), 2 * Matrix(1e-5 * I, D, D)\nsqrt_Σ₀ = cholesky(Σ₀).U\nnothing # hide","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"Next, generate an example state trajectory and according measurements.","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"ground_truth, data = simulate(A, Q, zeros(D), H, R, zeros(d), μ₀, Σ₀, 200, rng=MersenneTwister(3))\nnothing # hide","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"Compute the filtering posterior.","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"sol = [(μ₀, sqrt.(diag(Σ₀)))]\nfcache = SqrtKFCache(D, d)\nfcache.μ .= μ₀\ncopy!(fcache.Σ, sqrt_Σ₀)\nfor y in data\n    sqrt_kf_predict!(fcache, A, sqrt_Q)\n    sqrt_kf_correct!(fcache, H, sqrt_R, y)\n    push!(sol, (copy(fcache.μ), sqrt.(diag(Matrix(fcache.Σ)))))\nend\nnothing # hide","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"Finally, plot the results.","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"scatter([y[1] for y in data], [y[2] for y in data], label=\"Measurements\", markersize=2)\nplot!([y[1] for y in ground_truth], [y[2] for y in ground_truth], label=\"True Location\", linewidth=4, alpha=0.8)\nplot!(\n    [y[1] for (y, s) in sol], [y[2] for (y, s) in sol],\n    ribbon=(1.96 .* [s[1] for (y, s) in sol], 1.96 .* [s[2] for (y, s) in sol]),\n    label=\"Filter Estimate\", linewidth=4, alpha=0.8\n)\nsavefig(\"sqrt_kalman_filter_example.svg\")\nnothing # hide","category":"page"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"(Image: )","category":"page"},{"location":"examples/sqrt_kalman_filter/#References","page":"Square-root Kalman Filter","title":"References","text":"","category":"section"},{"location":"examples/sqrt_kalman_filter/","page":"Square-root Kalman Filter","title":"Square-root Kalman Filter","text":"[1] \"Bayesian Filtering and Smoothing\", Simo Särkka, Cambridge University Press, 2013.","category":"page"}]
}
