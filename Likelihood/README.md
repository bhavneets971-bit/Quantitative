# Likelihood Comparison of Observation Error Covariance for Yield Curves

## Overview

In this project, I study whether **measurement errors in U.S. Treasury yields are independent across maturities or correlated**.

I use a Kalman filter and innovation log-likelihood to compare two observation error models:
- a standard **diagonal** covariance model
- a **correlated** covariance model estimated using static Desroziers diagnostics

To ensure a fair comparison, I construct a nested likelihood test where the diagonal model appears as a special case of the correlated model.

---

## Modeling Idea (Intuition First)

I model the true yield curve as a latent state that evolves slowly over time. Observed yields equal the true curve plus measurement noise.

The key object of interest is the **observation error covariance matrix R**, which determines:
- how noisy each maturity is
- whether measurement errors move together across maturities

I compare two competing assumptions:
- **Diagonal R**: each maturity’s observation error is independent
- **Correlated R**: nearby maturities share common noise components

---

## Observation Error Models

### Diagonal R (Baseline)

The diagonal model assumes no cross-maturity dependence:
```
R = diag(σ₁², σ₂², ..., σₙ²)
```

This assumption is widely used but restrictive.

---

### Desrosiers-Based R (Correlated)

I estimate a correlated observation error covariance matrix using the Desrosiers diagnostic:
- I run a Kalman filter
- I collect innovation and analysis residuals
- I estimate cross-maturity error covariances from their empirical cross-products

This approach reveals the correlation structure of measurement noise.

---

## Why Innovation Log-Likelihood?

The Kalman filter produces **innovations**:
```
dₜ = yₜ − predicted observation
```

Under Gaussian assumptions:
```
dₜ ~ N(0, Sₜ),   where   Sₜ = Pₜ + R
```

The innovation log-likelihood measures how well a given covariance model explains the observed surprises in the data.

- Higher log-likelihood → better statistical fit
- Lower log-likelihood → poorer fit

---

## Nested Likelihood Comparison Using α

To compare models fairly, I define a one-parameter family:
```
R(α) = (1 − α) · diag(R_des) + α · R_des
```

Interpretation:
- `α = 0` → purely diagonal observation errors
- `α = 1` → fully correlated (Desrosiers) observation errors
- intermediate values → partial correlation

I evaluate the log-likelihood across several values of α and select the value that maximizes fit.

---

## What I Changed to Make the Comparison Valid

I initially observed that a naive comparison favored the diagonal model. That happened because the diagonal covariance was implicitly tuned to the data, while the Desrosiers covariance was diagnostic rather than likelihood-optimized.

To fix this, I:
1. Embedded the diagonal model inside the correlated model using α
2. Used the same Kalman filter settings for all models
3. Evaluated likelihood consistently across α
4. Used Cholesky factorization for numerical stability

These changes ensure that the likelihood comparison is meaningful.

---

## Results

The innovation log-likelihood increases monotonically as α increases.

**Best result:**
```
α = 1.0
```

This result shows that the fully correlated Desrosiers covariance explains the data much better than the diagonal alternative.

---

## Assumptions

- Linear Gaussian state-space model
- Random-walk dynamics for the latent yield curve
- Gaussian observation and process noise
- Time-invariant observation error covariance
- Fixed process noise covariance Q

---

## Limitations

- The Desrosiers covariance is diagnostic, not a maximum-likelihood estimate
- Q and R have not been jointly estimated
- out-of-sample validation has not been performed
- The state-space model is intentionally simple

---

## References

- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space*. Quarterly Journal of the Royal Meteorological Society.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.

---