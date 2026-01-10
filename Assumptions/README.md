# Assumptions & Limitations

This project focuses on **diagnosing model behavior and stability** using a Kalman filter framework applied to financial time series. The assumptions below are made deliberately and transparently, with the understanding that many are simplifications of real-world market dynamics.

---

## Key Assumptions

### 1. Linear Gaussian Model
- The system is modeled using a **linear state-space model with Gaussian noise**.
- This assumption enables analytical tractability and well-understood diagnostics.
- The model is **not assumed to be a perfect description of markets**; deviations are expected and informative.

---

### 2. Local Stationarity
- Financial time series are assumed to be **approximately stationary within rolling windows**, but not over the full sample.
- Model parameters are held constant within each window and allowed to evolve over time via re-estimation.
- Window length is treated as a tunable hyperparameter balancing stability and responsiveness.

---

### 3. Innovation Properties
- If the model is appropriate, Kalman filter innovations should be:
  - Zero-mean  
  - Serially uncorrelated  
  - Correctly scaled by their covariance
- These properties are **explicitly tested**, not assumed to always hold.

---

### 4. Covariance Estimation
- Sample-based estimates of process and observation covariances are assumed to be reasonable within each window.
- Estimation noise is expected, particularly for short windows or higher-dimensional observations.
- Eigenvalues and off-diagonal behavior are monitored to detect instability or misspecification.

---

### 5. Likelihood-Based Evaluation
- Gaussian log-likelihoods are used as a **relative comparison tool** (e.g. in-sample vs out-of-sample), not as an absolute measure of model correctness.
- Heavy tails and extreme events common in financial data may not be fully captured.

---

### 6. Numerical Stability
- Covariance matrices are assumed to be numerically well-conditioned and invertible.
- Diagnostics are used to identify degeneracy or instability rather than silently correcting it.

---

## Limitations

- The linear Gaussian assumption may fail during periods of market stress or regime change.
- Volatility clustering, jumps, and heavy tails are not explicitly modeled.
- Correlation structures are estimated empirically and may be noisy in small samples.
- The framework is diagnostic-focused and **not optimized for short-horizon prediction accuracy**.

---

## Directions for Improvement

Future work could strengthen the model and diagnostics by exploring:

- **Non-Gaussian innovations** (e.g. Student-t or robust filters)
- **Time-varying or regime-switching dynamics**
- **Factor-based observation models** to better capture cross-sectional structure
- **Regularization or shrinkage** for covariance estimation
- **Alternative scoring rules** beyond Gaussian likelihoods
- **Explicit numerical stabilization techniques** for near-singular covariances

---

## Summary

This project intentionally favors **simplicity, transparency, and diagnostics** over complexity.  
Its primary goal is to understand *when* and *why* model assumptions break, and to use that information to guide model trust and future refinement.
