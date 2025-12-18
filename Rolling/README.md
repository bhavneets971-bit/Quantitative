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

## Results Summary

Several clear and consistent patterns emerge from the rolling observation error variance estimates.

- **Strong maturity dependence**  
  Observation noise varies substantially across maturities. Short-term yields exhibit the
  highest variance, while longer maturities display lower and smoother noise profiles.
  This ordering remains stable across the sample.

- **Pronounced regime change at the short end**  
  Very short maturities, particularly at the front of the curve, show a marked decline in
  estimated observation noise over time. This shift occurs gradually and is consistent
  across nearby short maturities, suggesting a structural improvement in market liquidity
  or data quality rather than estimation noise.

- **Relative stability of the long end**  
  Longer-dated maturities exhibit comparatively low and slowly evolving observation
  variance. Changes over time are modest and occur smoothly, indicating a more stable
  measurement environment at long horizons.

- **Clear time variation in noise magnitude**  
  Observation error variance is not constant over the sample. Instead, it evolves
  gradually, with identifiable periods of higher and lower measurement uncertainty that
  affect multiple maturities simultaneously.

---

## Interpretation

The rolling estimates indicate that observation noise in yield curve data is both
maturity-dependent and time-varying. Noise is highest and most regime-sensitive at the
short end of the curve, while longer maturities display greater stability and lower overall
measurement uncertainty.

These results imply that observation errors are neither independent across maturities nor
stationary over long samples. Static diagonal observation error models fail to capture the
observed variation in noise magnitude and risk overstating confidence during noisier
periods. Allowing the observation error covariance to evolve over time provides a more
realistic description of measurement uncertainty and supports more stable filtering
behavior.

---

## Expected vs Observed Behavior

**Expected**
- Smooth evolution of observation noise over time  
- Higher variance at short maturities relative to the long end  
- Greater regime sensitivity at the front of the curve  

**Observed**
- Smooth and stable rolling variance estimates across all maturities  
- A clear and persistent maturity ordering in noise magnitude  
- Distinct regime shifts at short maturities, with comparatively stable behavior at longer
  horizons  

The close alignment between expected and observed behavior supports the validity of the
rolling Desrosiers implementation and suggests that the estimated time variation reflects
genuine changes in the measurement environment rather than numerical artifacts.

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