# Rolling Observation Error Covariance Estimation
## Spectral Diagnostics of Time-Varying Measurement Noise

---

## Overview

This project implements a rolling-window version of the **Desroziers innovation diagnostic**
to estimate a **time-varying observation error covariance matrix** within a Kalman
filtering framework. In addition to covariance and correlation analysis, I place
particular emphasis on **eigenvalue-based (spectral) diagnostics**, which provide a
compact and theoretically grounded summary of the structure and dimensionality of
observation error.

The empirical application focuses on U.S. Treasury yield curve data, though the
methodology is fully general and applicable to any linear Gaussian state-space model.

---

## Model Context

I work within a standard linear Gaussian state-space model.

State evolution:
```
x_t = F x_{t-1} + w_t ,   w_t ~ N(0, Q)
```

Observation model:
```
y_t = H x_t + v_t ,       v_t ~ N(0, R_t)
```

where:
- `x_t` is the latent (true) yield curve
- `y_t` is the observed yield curve
- `R_t` is the observation error covariance matrix, allowed to vary over time
- `Q` is the process noise covariance

The objective of this project is to **estimate and interpret the time variation and
structure of `R_t`**, rather than to assume it is fixed or diagonal.

---

## Desroziers Innovation Diagnostic

At each time step, the Kalman filter produces two residuals:

Innovation (forecast residual):
```
d_b(t) = y_t − H x_b(t)
```

Analysis residual (posterior residual):
```
d_a(t) = y_t − H x_a(t)
```

Under standard assumptions (linearity, unbiasedness, and near-optimal gain), the
observation error covariance satisfies:
```
R = E[ d_a(t) d_b(t)' ]
```

In practice, this expectation is approximated by sample averages. In this project, the
approximation is computed **over a rolling window**, producing a sequence of covariance
matrices `R_t`.

---

## Rolling-Window Estimation

For each time index `t`, I estimate:
```
R_t ≈ (1 / W) * Σ_{s = t−W}^{t−1} d_a(s) d_b(s)'
```

where `W` is a fixed rolling window length. This approach allows observation error
statistics to adapt gradually to changes in market structure and measurement quality.

---

## Numerical Safeguards

To ensure stability and interpretability:

**Symmetry**
```
R_t ← 0.5 * (R_t + R_t')
```

**Positive semi-definiteness**
Small or negative eigenvalues induced by finite-sample noise are clipped to a small
positive threshold. This guarantees that `R_t` remains a valid covariance matrix.

---

## Why Eigenvalues Matter

While covariance and correlation matrices describe **pairwise relationships**, eigenvalue
analysis addresses a different and more fundamental question:

> *How many independent sources of observation error are present, and how strong are they?*

Given a symmetric positive semi-definite matrix `R_t`, the eigendecomposition
```
R_t = U_t Λ_t U_t'
```
expresses observation error as a sum of orthogonal modes.

- Columns of `U_t` are **eigenvectors** (error modes)
- Diagonal entries of `Λ_t` are **eigenvalues** (variance carried by each mode)

Each eigenvalue therefore represents the variance of observation error along a
particular independent direction in maturity space.

---

## Interpretation of Eigenvalues

### 1. Dimensionality of Observation Error

If only a few eigenvalues are large while the rest are small, then observation error is
**effectively low-dimensional**, even if `R_t` is full rank.

In this application:
- The effective rank of `R_t` is approximately **three**
- Most observation uncertainty is captured by a small number of modes

This implies that yield curve measurement noise is structured rather than diffuse.

---

### 2. Dominant Error Modes

The leading eigenvalue typically explains more than half of total observation variance.
This indicates a **dominant common error component**, naturally interpreted as a
*level-like* measurement error affecting all maturities simultaneously.

Subsequent eigenvalues correspond to weaker but economically meaningful modes, often
associated with slope and curvature distortions.

---

### 3. Spectral Stability and Regime Change

Tracking eigenvalues over time provides a powerful diagnostic tool.

- Smooth eigenvalue evolution indicates a stable measurement environment
- Sharp eigenvalue spikes correspond to periods of market stress
- Changes in eigenvalue separation signal shifts in the structure of observation error

These diagnostics are far easier to interpret than raw covariance matrices.

---

### 4. Effective Rank

To summarize spectral information, I compute the **effective rank**:
```
r_eff = (Σ λ_i)^2 / Σ λ_i^2
```

This quantity measures how many eigenmodes contribute meaningfully to total variance.

- `r_eff ≈ 1`: a single dominant error mode
- `r_eff ≈ 2–3`: structured but low-dimensional error
- Large `r_eff`: diffuse, high-dimensional noise

In the yield curve application, `r_eff` remains consistently near three.

---

## Empirical Findings

- Observation error is **time-varying**, with sharp increases during known stress periods
- Errors are **strongly correlated across maturities**
- A small number of eigenmodes dominate total uncertainty
- Long maturities are consistently less noisy than short maturities

These findings are consistent with market microstructure effects and known liquidity
patterns.

---

## Modeling Implications

Eigenvalue diagnostics strongly reject static diagonal observation error models.

Ignoring correlation and low-rank structure forces the Kalman filter to misattribute
measurement noise to state dynamics, degrading inference and stability. Adaptive or
structured observation error models provide a more realistic representation of
measurement uncertainty.

---

## Limitations

- The Desroziers estimator is diagnostic rather than maximum-likelihood
- The rolling window length is fixed
- Process noise `Q` is not jointly estimated
- Structural interpretation of eigenmodes is qualitative

These limitations are acceptable given the diagnostic focus of the project.

---

## Conclusion

Eigenvalue-based diagnostics reveal that observation error in yield curve data is
structured, low-dimensional, and time-varying. Spectral analysis provides a compact and
theoretically grounded way to assess the stability and realism of observation error
models, complementing covariance and correlation-based views.

---

## References

- Desrosiers, J., and Ivanov, S. (2001). *Diagnosis and adaptive tuning of observation-error parameters in a variational assimilation.* Quarterly Journal of the Royal Meteorological Society. 
- Desrosiers, J., Berre, L., Chapnik, B., and Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space.* Quarterly Journal of the Royal Meteorological Society. 
- Anderson, B. D. O., and Moore, J. B. (1979). *Optimal Filtering.* Prentice Hall. 
- Durbin, J., and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods.*
- Horn, R. A., & Johnson, C. R. (2013). *Matrix Analysis.* Cambridge University Press.

