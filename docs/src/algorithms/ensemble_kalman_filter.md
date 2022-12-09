# Ensemble Kalman Filter

### References

- [1] Mandel, J. (2006). Efficient Implementation of the Ensemble Kalman Filter.
- [2] M. Katzfuss, J. R. Stroud, and C. K. Wikle, “Understanding the Ensemble Kalman Filter,” The American Statistician, vol. 70, no. 4, pp. 350–357,  2016.

### Methods

# EnKF Prediction
```@docs
enkf_predict
enkf_predict!
```

# EnKF Correction
```@docs
enkf_correct
enkf_correct!
```

### Observation-matrix-free version
```@docs
enkf_matrixfree_correct
enkf_matrixfree_correct!
```

# Auxiliary methods
### Compute moments from an ensemble
```@docs
ensemble_mean
ensemble_mean!
centered_ensemble
centered_ensemble!
ensemble_cov
ensemble_mean_cov
ensemble_mean_cov!
```

### Misc.
```@docs
PhDSE.A_HX_HA!
```
