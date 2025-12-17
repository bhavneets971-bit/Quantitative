# Desrosiers-Based Observation Error Estimation
## A Kalman Filter Application to Yield Curve Data

---

## 1. Project Overview

This project implements the **Desrosiers method** (Desrosiers & Ivanov, 2001; Desrosiers et al., 2005) to estimate **observation (measurement) error covariance matrices** using innovation-based diagnostics within a **Kalman filtering framework**.

The method is applied to **U.S. Treasury yield curve data**, treating observed yields as noisy measurements of an underlying latent “true” yield curve. The estimated covariance and correlation matrices characterize **correlated observation noise across maturities**, improving upon standard diagonal noise assumptions.

---

## 2. Signal vs Noise

- **True state**: latent, unobservable quantity (true yield curve)
- **Observation**: noisy measurement (quoted market yields)

Noise may be structured and correlated across maturities. Accurately estimating this structure is essential for reliable inference.

---

## 3. State–Space Model

### State evolution
x_t = F x_(t−1) + w_t , w_t ~ N(0, Q)

### Observation model
y_t = H x_t + v_t , v_t ~ N(0, R)

Where:
- x_t : latent true yield curve  
- y_t : observed yields  
- Q   : process noise covariance  
- R   : observation error covariance (unknown)

The goal of this project is to estimate **R**.

---

## 4. Kalman Filter Mechanics

At each time step the Kalman filter performs:

1. **Forecast**: predict state and uncertainty  
2. **Innovation**  
   d_b = y_t − H x_b  
3. **Update**: combine model and data using Kalman gain  
4. **Analysis residual**  
   d_a = y_t − H x_a  

---

## 5. Innovations vs Analysis Residuals

| Residual | Process Noise | Observation Noise |
|--------|---------------|------------------|
| Innovation (d_b) | Large | Large |
| Analysis residual (d_a) | Reduced | Large |

This asymmetry enables separation of observation noise.

---

## 6. Desrosiers Method

Core result:

R = E[ d_a · d_bᵀ ]

In practice:

R ≈ (1 / T) Σ d_a(t) d_b(t)ᵀ

This allows estimation of observation error covariance **without knowing the true state**.

---

## 7. Implementation Summary

1. Load and clean yield curve data  
2. Run Kalman filter  
3. Store innovations and analysis residuals  
4. Estimate R using Desrosiers diagnostics  
5. Enforce positive semi-definiteness  
6. Save covariance and correlation matrices  

Outputs:
- observation_error_covariance.csv  
- observation_error_correlation.csv  

---

## 8. Results Summary

- Strong short-end correlations (e.g., 3M–6M ≈ 0.8)
- Smooth decay of correlation with maturity separation
- Weak short–long coupling
- Positive variances across all maturities

---

## 9. Assumptions & Limitations

- Linear Gaussian dynamics
- Approximate stationarity
- Near-optimal Kalman gain
- Diagnostic (not causal) estimation

---

## 10. One-Sentence Summary

This project estimates correlated observation noise in a Kalman-filtered yield curve using innovation-based diagnostics, revealing structured measurement uncertainty beyond diagonal assumptions.
