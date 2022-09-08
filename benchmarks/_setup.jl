function kalman_setup(; D, d)
    Random.seed!(1234)

    # Dummy dynamics
    Φ = randn(D, D)
    _r_q = randn(D, D)
    Q = _r_q' * _r_q + diagm(0 => 1e-7 .* ones(D))
    u = randn(D)

    # Dummy measurements
    H = randn(d, D)
    _r_r = randn(d, d)
    R = _r_r' * _r_r + diagm(0 => 1e-7 .* ones(d))
    v = randn(d)
    y = randn(d)

    # Dummy initial conditions
    μ₀ = randn(D)
    _r_c = randn(D, D)
    Σ₀ = _r_c' * _r_c + diagm(0 => 1e-7 .* ones(D))

    return Φ, Q, u, H, R, v, y, μ₀, Σ₀
end
