# Desrosiers-Based Observation Error Estimation  
## Kalman Filtering Applied to Yield Curve Data

---

## Overview

This project applies the **Desrosiers innovation diagnostics** to estimate the **observation error covariance matrix** in a Kalman filter, using U.S. Treasury yield curve data.

Instead of assuming that measurement errors are independent across maturities, I estimate how observation noise is **structured and correlated** along the yield curve. The goal is practical: understand how much of the observed yield curve movement reflects shared measurement noise rather than true changes in the underlying curve.

---

## Signal vs. Noise (Intuition)

- **True state**: the latent, unobservable yield curve  
- **Observation**: quoted market yields, which contain noise  

That noise is not random maturity-by-maturity. Nearby maturities are quoted, interpolated, and traded together, so their errors tend to move together. This project explicitly measures that structure.

---

## State–Space Model

I use a standard linear Gaussian state-space model.

### State evolution
```
x_t = F x_{t−1} + w_t,     w_t ~ N(0, Q)
```

### Observation model
```
y_t = H x_t + v_t,        v_t ~ N(0, R)
```

Where:
- `x_t` is the latent (true) yield curve  
- `y_t` is the observed yield curve  
- `Q` controls how the true curve evolves  
- `R` is the observation error covariance (unknown)

The purpose of this project is to estimate **R**.

---

## Kalman Filter Residuals

At each time step, the Kalman filter produces two key residuals:

- **Innovation (forecast residual)**  
  ```
  d_b = y_t − H x_t^b
  ```

- **Analysis residual (post-update residual)**  
  ```
  d_a = y_t − H x_t^a
  ```

The innovation contains both process noise and observation noise.  
The analysis residual largely removes process noise but still reflects observation noise.

---

## Why This Matters

| Residual | Process Noise | Observation Noise |
|--------|---------------|------------------|
| Innovation (`d_b`) | High | High |
| Analysis residual (`d_a`) | Reduced | High |

This difference is exactly what makes the Desrosiers method work.

---

## Desrosiers Diagnostic

Under standard Kalman filter assumptions, observation error covariance satisfies:
```
R = E[d_a d_bᵀ]
```

In practice, I estimate:
```
R ≈ (1 / T) Σ d_a(t) d_b(t)ᵀ
```

This allows estimation of observation error covariance **without ever observing the true state**.

---

## Implementation Steps

1. Load and clean Treasury yield curve data  
2. Run a Kalman filter with simple dynamics  
3. Store innovations and analysis residuals  
4. Estimate R using the Desrosiers diagnostic  
5. Symmetrize and enforce positive definiteness  
6. Save covariance and correlation matrices  

Outputs:
- `observation_error_covariance.csv`
- `observation_error_correlation.csv`

---

## Results (Summary)

The estimated observation error structure shows:

- Strong correlations at the short end of the curve  
- Smooth decay of correlation with maturity distance  
- Weak interaction between very short and very long maturities  
- Positive, well-behaved noise variances across all maturities  

These patterns are consistent with market microstructure and curve construction practices.

---

## Assumptions and Limitations

- Linear Gaussian dynamics  
- Approximately stationary error statistics  
- Kalman gain close to optimal  
- Diagnostic estimation (not a causal or structural model)  

The results are meant for **diagnosis and validation**, not direct parameter inference.

---

## References

- Desroziers, G., & Ivanov, S. (2001). *Diagnosis and adaptive tuning of observation-error parameters in a variational assimilation*. Quarterly Journal of the Royal Meteorological Society.  
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space*. Quarterly Journal of the Royal Meteorological Society.  
- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.  
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.

---