# Observation Error Covariance Estimation and Validation  
### Innovation-Based Diagnostics with Kalman Filtering

---

## Motivation

Many estimation, filtering, and retrieval problems rely on accurate specification of the **observation error covariance matrix (R)**. In practice, R is often assumed to be diagonal for simplicity, implying independent measurement errors across channels or variables. However, this assumption is frequently violated in real-world systems where measurements may share common noise sources, calibration effects, or structural dependencies.

The goal of this project is to:
- **Estimate observation error covariance structures directly from data**, rather than assuming them a priori.
- **Validate whether structured (correlated) error models are statistically justified**, using likelihood-based diagnostics.
- Build a **methodologically sound foundation** for future applications in optimal estimation and machine-learning–based inference.

This repository explores these questions using **Kalman filtering and innovation diagnostics**, with methods transferable across domains such as remote sensing, finance, and data assimilation.

---

## The Desrosiers Innovation Diagnostic

The core methodology implemented here is based on the **Desrosiers method** (Desrosiers & Ivanov, 2001; Desrosiers et al., 2005), an innovation-based diagnostic technique for estimating observation error covariance.

### Key idea

In a Kalman filter, two residuals are naturally produced at each time step:

- **Innovation**  
  \[
  d^b_t = y_t - H x^b_t
  \]
  (difference between observation and forecast)

- **Analysis residual**  
  \[
  d^a_t = y_t - H x^a_t
  \]
  (difference between observation and updated state)

Under standard Kalman filter assumptions (linearity, unbiased errors, near-optimal gain), the observation error covariance can be estimated as:

\[
R \approx \mathbb{E}[ d^a_t (d^b_t)^T ]
\]

This allows estimation of **correlated observation noise** without direct access to the true state.

---

## Why Validation Is Necessary

While the Desrosiers method can reveal rich error correlation structures, **estimating R does not guarantee that it improves model performance**. A structured covariance may:
- Overfit finite-sample noise
- Be inconsistent with the assumed state-space model
- Degrade probabilistic performance when used in filtering

Therefore, this project emphasizes **likelihood-based validation**, comparing different R models under the same Kalman filter to assess their statistical optimality.

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
