# Desrosiers-Based Observation Error Estimation

## A Kalman Filter Application to Yield Curve Data

------------------------------------------------------------------------

## 1. Project Overview

This project implements the **Desrosiers method** (Desrosiers & Ivanov,
2001; Desrosiers et al., 2005) to estimate **observation (measurement)
error covariance matrices** using innovation-based diagnostics within a
**Kalman filtering framework**.

The method is applied to **U.S. Treasury yield curve data**, treating
observed yields as noisy measurements of an underlying latent "true"
yield curve. The estimated covariance and correlation matrices
characterize **correlated observation noise across maturities**,
improving upon standard diagonal noise assumptions.

Although demonstrated on financial data, the methodology is directly
transferable to: - Financial engineering and quant research - Risk and
factor models - State-space models - Optimal estimation (OE) - Machine
learning and ANN-based retrievals - Remote sensing and data assimilation

------------------------------------------------------------------------

## 2. Signal vs Noise

The key distinction underlying this project is:

-   **True state**: latent, unobservable quantity (true yield curve)
-   **Observation**: noisy measurement (quoted market yields)

Observation noise can be structured and correlated across maturities.
Accurately estimating this structure is essential for reliable
inference.

------------------------------------------------------------------------

## 3. State--Space Model

### State evolution

\[ x_t = F x\_{t-1} + w_t, `\quad `{=tex}w_t
`\sim `{=tex}`\mathcal{N}`{=tex}(0, Q) \]

### Observation model

\[ y_t = H x_t + v_t, `\quad `{=tex}v_t
`\sim `{=tex}`\mathcal{N}`{=tex}(0, R) \]

Where: - (x_t): latent true yield curve\
- (y_t): observed yields\
- (Q): process noise covariance\
- (R): observation error covariance (unknown)

The goal of this project is to estimate (R).

------------------------------------------------------------------------

## 4. Kalman Filter Mechanics

At each time step the Kalman filter performs:

1.  **Forecast**: predict the state and uncertainty\
2.  **Innovation**: \[ d_t\^b = y_t - H x_t\^b \]
3.  **Update**: combine model and data using the Kalman gain\
4.  **Analysis residual**: \[ d_t\^a = y_t - H x_t\^a \]

------------------------------------------------------------------------

## 5. Innovations vs Analysis Residuals

These residuals contain different mixtures of noise:

  Residual                   Process Noise   Observation Noise
  -------------------------- --------------- -------------------
  Innovation (d\^b)          Large           Large
  Analysis residual (d\^a)   Reduced         Large

This asymmetry enables separation of observation noise.

------------------------------------------------------------------------

## 6. Desrosiers Method

Under standard Kalman filter assumptions:

\[ R = `\mathbb{E}`{=tex}\[ d_t\^a (d_t^b)^T \] \]

In practice:

\[ R `\approx `{=tex}`\frac{1}{T}`{=tex} `\sum`{=tex}\_{t=1}\^T d_t\^a
(d_t^b)^T \]

This allows estimation of observation error covariance **without knowing
the true state**.

------------------------------------------------------------------------

## 7. Implementation Summary

1.  Load and clean yield curve data\
2.  Run Kalman filter\
3.  Store innovations and analysis residuals\
4.  Estimate observation error covariance via Desrosiers diagnostics\
5.  Enforce positive semi-definiteness\
6.  Save covariance and correlation matrices

Outputs: - `observation_error_covariance.csv` -
`observation_error_correlation.csv`

------------------------------------------------------------------------

## 8. Results Summary

-   Strong short-end correlations (e.g., 3M--6M ≈ 0.8)
-   Smooth decay of correlation with maturity separation
-   Weak short--long coupling
-   Positive variances across all maturities

Results are statistically stable and economically interpretable.

------------------------------------------------------------------------

## 9. Assumptions & Limitations

-   Linear Gaussian dynamics
-   Approximate stationarity
-   Near-optimal Kalman gain
-   Diagnostic (not causal) estimation

------------------------------------------------------------------------

## 10. One-Sentence Summary

*This project estimates correlated observation noise in a
Kalman-filtered yield curve using innovation-based diagnostics,
revealing structured measurement uncertainty beyond diagonal
assumptions.*
