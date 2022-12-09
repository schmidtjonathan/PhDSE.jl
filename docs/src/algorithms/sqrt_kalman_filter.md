# Square-root Kalman Filter

# Note regarding Sqrt-Kalman filter implementation
The present implementation assumes **right** matrix-square roots of the involved covariance matrices.
This means that a PSD covariance matrix $\mathrm{P}$ is decomposed as

$$\mathrm{P} = \mathrm{U}^\top \mathrm{U}.$$

This is in contrast to assuming a decomposition $\mathrm{P} = \mathrm{L} \mathrm{L}^\top$, as is also often the convention.

### References

- [1] Mohinder S Grewal and Angus P Andrews. Kalman filtering: Theory and Practice with MATLAB. John Wiley & Sons, 2014.
- [2] Krämer, N., & Hennig, P. (2020). Stable implementation of probabilistic ODE solvers. arXiv:2012.10106.

### Methods

# Sqrt-KF Prediction
```@docs
sqrt_kf_predict
sqrt_kf_predict!
```

# Sqrt-KF Correction
```@docs
sqrt_kf_correct
sqrt_kf_correct!
```