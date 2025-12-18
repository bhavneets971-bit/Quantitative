# Observation Error Covariance Estimation and Validation  
### Innovation-Based Diagnostics with Kalman Filtering

---
## Practical Relevance for Trading and Finance

This project is not a trading strategy by itself, and it is not intended to generate direct buy or sell signals. Its value lies one level deeper, in how market data is interpreted before any trading decision is made.

In fixed-income markets, many trading signals are derived from small changes in the yield curve (level, slope, curvature). If observation noise is mis-specified, especially if correlated noise is treated as independent, models tend to overreact to routine market noise. This leads to false signals, unstable factor estimates, and excessive trading.

By explicitly modelling correlated observation errors, this framework helps distinguish genuine yield curve movements from shared measurement noise. In practice, this results in smoother state estimates, more reliable factor extraction, and more realistic assessments of uncertainty. These improvements matter for position sizing, risk management, and the robustness of backtests.

In short, the contribution of this work is not to predict markets, but to reduce overconfidence and noise-driven decisions, an effect that can materially improve risk-adjusted performance when embedded inside larger trading or portfolio construction systems.

## Motivation

Accurate specification of the **observation error covariance matrix (R)** is central to filtering, estimation, and retrieval problems. In practice, R is often assumed to be diagonal for convenience, implying independent measurement errors across variables or channels.

This assumption is rarely realistic. Measurements frequently share common noise sources, interpolation artifacts, market microstructure effects, or calibration errors, all of which introduce **correlated observation noise**.

The purpose of this project is threefold:

- Estimate observation error covariance structures **directly from data**, rather than imposing diagonal assumptions.
- Test whether correlated error models are **statistically justified**, using likelihood-based validation.
- Establish a clean and extensible framework suitable for downstream applications in filtering, optimal estimation, and learning-based models.

Although the empirical application uses U.S. Treasury yield data, the methodology is general and applies equally to problems in finance, remote sensing, and data assimilation.

---

## The Desrosiers Innovation Diagnostic

The core methodology follows the **Desrosiers innovation diagnostic**, originally developed in the data assimilation literature.

### Core idea

A Kalman filter naturally produces two residuals at each time step:

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

In practice, the expectation is replaced by a time average. This provides an estimate of the **observation error covariance matrix**, including cross-correlations, without direct access to the true state.

---

## Why Validation Is Necessary

Estimating a structured covariance matrix does not guarantee improved model performance.

A correlated R may:
- Capture finite-sample noise rather than true structure
- Be inconsistent with the assumed state-space dynamics
- Reduce probabilistic performance when used inside a filter

For this reason, estimation and validation are treated as separate steps. After estimating R using Desrosiers diagnostics, I evaluate whether the resulting covariance improves model fit by comparing **innovation log-likelihoods** under different observation error models.

---

## Repository Structure

```text
.
├── static_desrosiers_error/
│   ├── estimate_static_R.py
│   └── README.md
│
├── likelihood/
│   ├── likelihood_validation.py
│   └── README.md
│
├── data/
│   └── yield-curve-rates-1990-2024.csv
│
└── README.md
```

---

## References

- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.
- Desroziers, G., & Ivanov, S. (2001). *Diagnosis and adaptive tuning of observation-error parameters in a variational assimilation*. Quarterly Journal of the Royal Meteorological Society.
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space*. Quarterly Journal of the Royal Meteorological Society.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
