# Observation Error Covariance Estimation and Validation  
### Innovation-Based Diagnostics with Kalman Filtering

---

## Executive Summary

This repository implements a **research-grade diagnostic and validation framework** for estimating and validating **correlated observation error covariance matrices** within linear Gaussian state-space models using Kalman filtering.

The project focuses on *model reliability rather than prediction*, addressing a common but under-examined source of failure in quantitative finance models: **mis-specified observation noise**. By replacing ad-hoc diagonal assumptions with data-driven covariance estimation and likelihood-based validation, the framework improves uncertainty quantification, state estimation stability, and downstream decision robustness.

This work is designed to be suitable for **quantitative research, model risk review, and integration into larger trading, risk, or portfolio construction systems**, and was developed as part of preparation for **MFE / MS in Quantitative Finance programs**.

---

## Practical Relevance for Trading and Finance

This project is **not** a trading strategy and does not produce buy or sell signals. Its value lies one layer deeper: in how market data is **filtered, interpreted, and trusted** *before* any trading or portfolio decision is made.

In fixed-income markets, economically meaningful signals often correspond to small changes in the yield curve (level, slope, curvature). When observation noise is mis-specified—particularly when **correlated measurement noise is incorrectly treated as independent**—state-space models overreact to routine market fluctuations. This leads to false signals, unstable factor estimates, excessive turnover, and misleading backtests.

By explicitly modeling **correlated observation errors**, this framework separates genuine yield curve movements from shared measurement noise. In practice, this yields smoother state estimates, more stable factor extraction, and more realistic uncertainty estimates. These improvements directly impact **position sizing, risk management, and downstream model robustness**.

The contribution of this work is not alpha generation. It is the **reduction of noise-driven overconfidence**, which can materially improve risk-adjusted performance when embedded inside larger quantitative systems.

---

## Motivation

Accurate specification of the **observation error covariance matrix (R)** is central to Kalman filtering and state-space modeling. In applied financial work, R is often assumed to be diagonal for convenience, implying independent measurement errors across maturities or variables.

This assumption is rarely realistic. Financial measurements frequently share common noise sources, including:

- Market microstructure effects  
- Quoting and interpolation artifacts  
- Liquidity conditions  
- Shared data construction and smoothing procedures  

These mechanisms naturally induce **correlated observation noise**.

The goals of this project are therefore:

- To estimate observation error covariance structures **directly from data**, rather than imposing diagonal assumptions  
- To test whether correlated error models are **statistically justified**, using likelihood-based validation  
- To build a clean, auditable pipeline suitable for reuse inside larger filtering, estimation, or learning systems  

Although the empirical application uses U.S. Treasury yields, the methodology is general and applies broadly across finance, signal processing, and data assimilation.

---

## Mathematical Framework

### State-Space Model

The system is modeled as a linear Gaussian state-space process:

State equation:
```
x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
```

Observation equation:
```
y_t = H x_t + v_t,       v_t ~ N(0, R)
```

where:
- `x_t` represents latent states (e.g., yield curve factors)
- `y_t` denotes observed yields
- `Q` is the process noise covariance
- `R` is the observation noise covariance

The focus of this project is the **estimation and validation of R**.

---

## The Desroziers Innovation Diagnostic

The core methodology follows the **Desroziers innovation diagnostic**, originally developed in the data assimilation literature.

A Kalman filter produces two residuals at each time step:

- **Innovation (forecast residual)**  
```
dᵇ_t = y_t − H xᵇ_t
```

- **Analysis residual (post-update residual)**  
```
dᵃ_t = y_t − H xᵃ_t
```

Under standard Kalman filter assumptions—linearity, unbiased errors, and a near-optimal Kalman gain—these residuals satisfy:

```
R = E[dᵃ dᵇᵀ]
```

In practice, the expectation is approximated using time averages, yielding an estimate of the **full observation error covariance matrix**, including cross-correlations, without requiring access to the true latent state.

---

## Why Validation Is Necessary

Estimating a structured covariance matrix does not automatically improve model performance.

A correlated observation error model may:
- Capture finite-sample noise rather than persistent structure  
- Be inconsistent with assumed state dynamics  
- Degrade probabilistic performance when embedded inside a filter  

For this reason, estimation and validation are treated as **distinct steps**. After estimating R using Desroziers diagnostics, I evaluate whether the resulting covariance improves model fit by comparing **innovation log-likelihoods** across competing observation error specifications.

---

## End-to-End Pipeline

This repository implements a full diagnostic and validation pipeline:

1. **Process noise (Q) diagnostics**
   - Empirical anchoring using yield changes  
   - Calibration via innovation whitening  

2. **Static observation error estimation**
   - Desroziers covariance estimation  
   - Positive-definiteness enforcement  
   - Correlation structure analysis  

3. **Rolling observation error estimation**
   - Time-varying R_t via rolling windows  
   - Detection of regime shifts and stress periods  

4. **Likelihood-based validation**
   - Diagonal vs static full vs rolling full R  
   - Common-sample likelihood comparison  

5. **Model health monitoring**
   - Cross-checks across all saved artifacts  
   - Automated PASS / WARN / FAIL verdicts  
   - Diagnostic plots for human inspection  

Estimation, validation, and diagnostics are intentionally separated to avoid circular reasoning and silent failures.

---

## Repository Structure

```
.
├── data/
│   └── yield-curve-rates-1990-2024.csv
├── Q_tuning/
│   ├── Q_tuning.py
│   └── README.md
├── static/
│   ├── Desrosiers.py
│   └── README.md
├── rolling/
│   ├── Rolling.py
│   ├── plot.py
│   ├── windowsize_tuning.py
│   └── README.md
├── likelihood/
│   ├── is_validation.py
│   ├── oos_validation.py
│   └── README.md
├── health/
│   ├── model_health.py
│   └── README.md
└── README.md
```

---

## Interpretation of Results

The rolling observation error variance exhibits:

- Low and stable values during calm market regimes  
- Large, temporary spikes during stress periods (e.g., 2008, 2020, 2022)  
- No numerical explosions or degeneracy  

This behavior is economically meaningful and reflects changing market conditions and data reliability rather than model instability.

Likelihood comparisons consistently show:

```
loglik(diagonal R) < loglik(static full R) < loglik(rolling full R)
```

This confirms that modeling **correlated and time-varying observation noise** materially improves probabilistic fit.

---

## What This Project Is and Is Not

**This project is:**
- A rigorous diagnostic and validation framework  
- Suitable for research, model risk review, and downstream integration  
- Designed to detect overconfidence and silent failures  

**This project is not:**
- A trading signal generator  
- A standalone forecasting model  
- A black-box machine learning system  

---

## References

- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.  
- Desroziers, G., & Ivanov, S. (2001). *Diagnosis and adaptive tuning of observation-error parameters in a variational assimilation*. QJRMS.  
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics in observation space*. QJRMS.  
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.

---

## Author

Bhavneets971  
Aspiring Financial Engineer | Quantitative Research  
bhavneet.bhatia@dal.ca