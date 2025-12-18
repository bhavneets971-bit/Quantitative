# Rolling Observation Error Covariance Estimation
## Desrosiers Innovation Diagnostics with a Rolling Window

---

## Overview

This module implements a rolling-window version of the **Desrosiers innovation diagnostic**
to estimate a **time-varying observation error covariance matrix** within a Kalman
filtering framework.

Rather than assuming that observation noise is constant over long samples, the rolling
approach allows the error structure to adapt gradually to changing market conditions,
liquidity regimes, and measurement environments. The empirical application focuses on
U.S. Treasury yield curve data, though the methodology is fully general.

The main output is a time series of observation error covariance matrices (R_t), stored in
a single long-format CSV file to facilitate inspection and analysis.

---

## Model Context

A standard linear Gaussian state-space model is assumed.

State evolution:
```
x_t = F x_{t-1} + w_t ,   w_t ~ N(0, Q)
```

Observation model:
```
y_t = H x_t + v_t ,       v_t ~ N(0, R_t)
```

where:
- x_t represents the latent (true) yield curve
- y_t denotes observed market yields
- R_t is the observation error covariance, allowed to vary over time
- Q is the process noise covariance

The objective of this module is to estimate R_t using innovation-based diagnostics.

---

## The Desrosiers Diagnostic

At each time step, the Kalman filter produces two residuals:

Innovation (forecast residual):
```
d_b(t) = y_t - H x_b(t)
```

Analysis residual (posterior residual):
```
d_a(t) = y_t - H x_a(t)
```

Under standard assumptions—linearity, unbiased errors, and a near-optimal Kalman gain—the
observation error covariance satisfies the relation:
```
R = E[ d_a(t) d_b(t)' ]
```

In practice, this expectation is approximated using sample averages. In the rolling
implementation, the approximation is computed over a fixed-length window rather than
over the full sample.

---

## Rolling-Window Estimation

For each time index t, the observation error covariance is estimated as:
```
R_t ≈ (1 / W) * sum_{s = t-W to t-1} d_a(s) d_b(s)'
```

where W denotes the rolling window length.

This procedure produces a sequence of covariance matrices that evolve smoothly over time,
reflecting gradual changes in the measurement environment rather than abrupt shifts.

---

## Numerical Safeguards

Two constraints are enforced to ensure numerical stability.

Symmetry:
```
R_t ← 0.5 * (R_t + R_t')
```

Positive definiteness:
Small or negative eigenvalues arising from finite-sample effects are clipped to a small
positive threshold. This guarantees that R_t remains a valid covariance matrix and can be
safely used in filtering and likelihood evaluation.

---

## Output Format

All rolling covariance matrices are stored in **long (tidy) format**, with one row per
covariance entry and time index.

| Column | Description |
|------|-------------|
| time_index | Rolling window index |
| maturity_i | First maturity |
| maturity_j | Second maturity |
| covariance | Estimated covariance value |

This format supports straightforward filtering, aggregation, and visualization.

---

## Key Empirical Findings

### 1. Strong Time Variation in Observation Noise

Observation error variance is clearly non-stationary over the sample period.

- **Early 1990s to early 2000s**  
  Variance levels are moderate and relatively smooth across maturities, with
  consistently higher noise at the short end of the curve.

- **Mid-2000s surge (≈2004–2007)**  
  A pronounced increase in observation noise appears across all maturities,
  especially for short and intermediate tenors (3M–2Y). Variance levels rise
  sharply relative to surrounding periods, consistent with changing market
  structure and elevated uncertainty.

- **Global Financial Crisis (≈2008–2009)**  
  Extremely sharp, short-lived variance spikes are observed, particularly at the
  short end. These reflect severe market dislocations, liquidity effects, and
  instability in observed yields.

- **Post-2010 low-noise regime**  
  From roughly 2010 to 2019, observation noise collapses to unusually low levels
  across the curve, indicating exceptionally clean measurement conditions.

- **COVID and post-COVID period (2020–2023)**  
  Observation noise increases again, led by short and intermediate maturities.
  While less extreme than during the GFC, the increase is clearly visible and
  economically meaningful.

---

### 2. Strong Maturity Dependence

Across all regimes, observation noise exhibits a clear term-structure:

- Short maturities (3M–1Y) consistently display the highest variances and the
  largest regime shifts.
- Long maturities (10Y–30Y) remain comparatively stable, even during stress
  periods.

This behavior is consistent with microstructure effects, liquidity variation,
and policy sensitivity being concentrated at the short end of the curve.

---

### 3. Abrupt Regime Transitions Reflect Structural Change

The sharp variance jumps observed around 2006–2009 are synchronized across
maturities and persist across rolling windows. This behavior is inconsistent
with numerical instability and instead reflects genuine deterioration in the
signal-to-noise ratio of observed yields during periods of market stress.

---

## Modeling Implications

These results directly challenge the validity of static diagonal observation
error models:

- Observation noise is time-varying
- Measurement uncertainty differs substantially across maturities
- Short-end noise dominates during stress periods

Using a fixed diagonal observation error covariance forces the Kalman filter to
misattribute measurement noise to state dynamics, degrading inference and
stability. Rolling or adaptive covariance models provide a more realistic
representation of measurement uncertainty.

---

## Expected vs. Observed Behavior

**Expected**
- Higher noise at short maturities
- Time variation in observation uncertainty
- Elevated noise during known stress periods

**Observed**
- All expected patterns are clearly present
- Stress periods produce sharp, coordinated variance spikes
- Long maturities remain comparatively stable
- Post-crisis regimes exhibit unusually low observation noise

The close alignment between expectation and observation supports both the rolling
Desrosiers implementation and its interpretation as genuine observation error.

---

## Conclusion

Rolling observation error variance estimates reveal that yield curve measurement
noise is structured, time-varying, and maturity-dependent. These features are
economically interpretable and statistically meaningful, providing strong
motivation for abandoning static diagonal observation error assumptions in favor
of adaptive covariance modeling.

---

## Limitations

- The Desrosiers estimator is diagnostic rather than maximum-likelihood
- The rolling window length is fixed and not optimized
- Process noise Q is not jointly estimated with R_t
- Results depend on the chosen Kalman filter specification

These limitations are acceptable for diagnostic and exploratory analysis.

---

## Practical Relevance

Although this module does not generate trading signals directly, it improves the quality
of state estimation by preventing overconfidence in noisy observations. In fixed-income
applications, this leads to more reliable factor estimates, more realistic uncertainty
quantification, and greater robustness in downstream modeling and decision-making.

---

## References

- Desrosiers, J., and Ivanov, S. (2001). Diagnosis and adaptive tuning of observation-error
  parameters in a variational assimilation. Quarterly Journal of the Royal Meteorological Society.
- Desrosiers, J., Berre, L., Chapnik, B., and Poli, P. (2005). Diagnosis of observation,
  background and analysis-error statistics in observation space. Quarterly Journal of the
  Royal Meteorological Society.
- Anderson, B. D. O., and Moore, J. B. (1979). Optimal Filtering. Prentice Hall.
- Durbin, J., and Koopman, S. J. (2012). Time Series Analysis by State Space Methods.

---