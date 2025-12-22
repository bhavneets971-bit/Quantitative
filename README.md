# Observation Error Covariance Estimation and Validation  
### Innovation-Based Diagnostics with Kalman Filtering

---

## Practical Relevance for Trading and Finance

This project is not a trading strategy, and it is not intended to generate direct buy or sell signals. Its value sits one level deeper, in how market data is interpreted *before* any trading or portfolio decision is made.

In fixed-income markets, many signals are driven by relatively small changes in the yield curve (level, slope, curvature). If observation noise is mis-specified — especially if correlated noise is treated as independent — models tend to overreact to routine market fluctuations. This leads to false signals, unstable factor estimates, excessive turnover, and misleading backtests.

By explicitly modelling **correlated observation errors**, this framework helps separate genuine yield curve movements from shared measurement noise. In practice, this produces smoother state estimates, more stable factor extraction, and more realistic uncertainty estimates. These improvements matter directly for position sizing, risk management, and the robustness of downstream models.

The contribution of this work is not market prediction. It is the reduction of overconfidence and noise-driven decisions, which can materially improve risk-adjusted performance when embedded inside larger trading, risk, or portfolio construction systems.

---

## Motivation

Accurate specification of the **observation error covariance matrix (R)** is central to filtering and state-space modeling. In practice, R is often assumed to be diagonal for convenience, implying independent measurement errors across maturities or variables.

This assumption is rarely realistic. Financial measurements often share common noise sources:
- Market microstructure effects  
- Quoting and interpolation artifacts  
- Liquidity conditions  
- Shared data construction and smoothing procedures  

All of these introduce **correlated observation noise**.

The goals of this project are therefore:

- To estimate observation error covariance structures **directly from data**, rather than imposing diagonal assumptions.
- To test whether correlated error models are **statistically justified**, using likelihood-based validation.
- To build a clean, auditable pipeline suitable for reuse inside larger filtering, estimation, or learning systems.

Although the empirical application uses U.S. Treasury yields, the methodology is general and applies to many problems in finance, signal processing, and data assimilation.

---

## The Desroziers Innovation Diagnostic

The core methodology follows the **Desroziers innovation diagnostic**, originally developed in the data assimilation literature.

### Core idea

A Kalman filter produces two residuals at each time step:

- **Innovation (forecast residual)**  
  ```
  dᵇₜ = yₜ − H xᵇₜ
  ```

- **Analysis residual (post-update residual)**  
  ```
  dᵃₜ = yₜ − H xᵃₜ
  ```

Under standard Kalman filter assumptions (linearity, unbiased errors, and a near-optimal Kalman gain), these residuals satisfy:

```
R = E[dᵃ dᵇᵀ]
```

In practice, the expectation is replaced by a time average. This provides an estimate of the **full observation error covariance matrix**, including cross-correlations, without requiring access to the true latent state.

---

## Why Validation Is Necessary

Estimating a structured covariance matrix does not automatically improve model performance.

A correlated R may:
- Capture finite-sample noise rather than true structure  
- Be inconsistent with the assumed state dynamics  
- Reduce probabilistic performance when used inside a filter  

For this reason, estimation and validation are treated as separate steps. After estimating R using Desroziers diagnostics, I explicitly test whether the resulting covariance improves model fit by comparing **innovation log-likelihoods** across competing observation error models.

---

## End-to-End Pipeline

This repository implements a full diagnostic and validation pipeline:

1. **Q diagnostics**
   - Empirical anchoring of process noise using yield changes  
   - Calibration via innovation whitening  

2. **Static observation error estimation**
   - Desroziers covariance estimation  
   - Positive-definiteness enforcement  
   - Correlation structure analysis  

3. **Rolling observation error estimation**
   - Time-varying Rₜ using rolling windows  
   - Detection of regime and stress periods  

4. **Likelihood-based validation**
   - Diagonal vs static full vs rolling full R  
   - Common-sample likelihood comparison  

5. **Model health monitoring**
   - Cross-checks across all saved artifacts  
   - Automated PASS / WARN / FAIL verdict  
   - Diagnostic plots for human inspection  

Each stage produces explicit outputs that are consumed by later stages. Estimation, validation, and diagnostics are intentionally separated to avoid circular reasoning.

---

## Repository Structure

```text
.
├── data/
│   └── yield-curve-rates-1990-2024.csv
│
├── diagnostics/
│   └── q_diagnostics.py
│
├── static/
│   ├── estimate_static_R.py
│   └── observation_error_covariance.csv
│
├── rolling/
│   ├── rolling_desrosiers_R.py
│   └── rolling_R_all.csv
│
├── likelihood/
│   ├── likelihood_validation.py
│   └── likelihood_summary.csv
│
├── health/
│   ├── model_health.py
│   └── plots/
│       └── rolling_R_trace.png
│
└── README.md
```

---

## Interpretation of Results

The rolling observation error variance exhibits:
- Low, stable values in calm market regimes  
- Large, temporary spikes during stress periods (e.g. 2008, 2020, 2022)  
- No numerical explosions or collapse  

This behavior is expected and economically meaningful. It reflects changes in market conditions and data reliability, not model failure.

Likelihood results consistently show:

```
loglik(diagonal R) < loglik(static full R) < loglik(rolling full R)
```

This confirms that modelling correlated and time-varying observation noise materially improves probabilistic fit.

---

## What This Project Is and Is Not

**This project is:**
- A rigorous diagnostic and validation framework  
- Suitable for research, model risk review, and downstream integration  
- Designed to catch silent failures and overconfidence  

**This project is not:**
- A trading signal generator  
- A forecasting model by itself  
- A black-box machine learning system  

---

## References

- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.  
- Desroziers, G., & Ivanov, S. (2001). *Diagnosis and adaptive tuning of observation-error parameters in a variational assimilation*. QJRMS.  
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space*. QJRMS.  
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
