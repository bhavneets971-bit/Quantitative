# Kalman Filter Innovation-Based Observation Error Diagnostics -- Model Health Report

## Overview

This project implements a **fully audited Kalman filtering pipeline**
for U.S. Treasury yield curve data (1990--2024), with a strong focus on
**diagnostics, validation, and model health monitoring**.

The core objective is to **estimate, validate, and monitor**:

-   Process noise covariance **Q**
-   Observation error covariance **R** (static and rolling)
-   Likelihood-based model comparisons
-   End-to-end internal consistency

------------------------------------------------------------------------

## Model Framework

I assume a linear Gaussian state-space model:

### State evolution

    x_t = x_{t-1} + w_t ,   w_t ~ N(0, Q)

### Observation equation

    y_t = x_t + v_t ,       v_t ~ N(0, R)

Where: - `x_t` = latent true yield curve - `y_t` = observed market
yields - `Q` = process noise covariance - `R` = observation error
covariance

Both `Q` and `R` are **empirically diagnosed**, not arbitrarily assumed.

------------------------------------------------------------------------

## Pipeline Components

### 1. Q Diagnostics

-   Empirical anchor: `Cov(Δy)`
-   Q scale selected via **innovation whitening**
-   Diagnostic target:


```
    E[||z_t||] ≈ √n
```

------------------------------------------------------------------------

### 2. Static Desroziers R Estimation

-   Uses innovation × analysis residual cross-products
-   Enforces symmetry and positive definiteness
-   Produces full covariance and correlation matrices

Key result: \> Observation errors are **strongly correlated across
neighboring maturities**, especially at the short end.

------------------------------------------------------------------------

### 3. Rolling Desroziers R Estimation

-   Rolling window: 252 trading days
-   Captures **time-varying observation noise**
-   Detects regime changes and market stress

------------------------------------------------------------------------

### 4. Likelihood Validation

Models compared under identical Q: - Diagonal R - Static full R -
Rolling full R

Result:

    loglik(diagonal) < loglik(static) < loglik(rolling)

This confirms that **correlated and time-varying observation noise
materially improves model fit**.

------------------------------------------------------------------------

## Model Health Monitor

A dedicated health monitor (`health/model_health.py`) consumes all saved
files and performs:

-   Whitening sanity checks
-   Positive-definiteness checks
-   Stability checks
-   Likelihood ordering checks
-   Cross-script consistency checks

### Verdict Levels

-   **PASS** -- Model fully stable
-   **WARN** -- Model stable, but regime-dependent behavior detected
-   **FAIL** -- Structural inconsistency (do not use)

------------------------------------------------------------------------

## Rolling R Diagnostic Plot

The health monitor automatically generates:

    health/plots/rolling_R_trace.png

### Interpretation

The plot shows **trace(R_t)** over time:

-   Low, stable noise in calm regimes
-   Large spikes during:
    -   2007--2009 Global Financial Crisis
    -   2020 COVID shock
    -   2022--2023 rate hiking cycle

This behavior is **economically correct** and indicates a
**well-calibrated diagnostic system**, not model failure.

The resulting **WARN verdict** reflects **non-stationary observation
noise**, which is expected in real financial markets.

------------------------------------------------------------------------

## Final Assessment

**The model is statistically coherent, economically interpretable, and
internally consistent.**

The health monitor confirms: - Correct Q calibration - Valid Desroziers
diagnostics - Meaningful rolling R dynamics - Robust likelihood
improvements

------------------------------------------------------------------------
