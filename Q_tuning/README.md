# Q Diagnostics for Kalman Filtering

## Purpose

This module performs **process-noise (Q) diagnostics** for a Kalman filter applied to U.S. Treasury yield curve data.

The objective is **not model comparison**, but **model calibration**: to determine a reasonable scale for the process noise covariance matrix Q before any likelihood-based evaluation of observation error models (R).

Correct calibration of Q is essential. If Q is poorly chosen, likelihood comparisons of R become misleading and unstable.

---

## Conceptual Background

In the state-space model

xₜ = xₜ₋₁ + wₜ , wₜ ~ N(0, Q)

Q represents **model uncertainty**: how much the latent “true” yield curve is allowed to move from one day to the next independently of measurement noise.

- Small Q → model is overly rigid  
- Large Q → model is overly flexible  
- Well-calibrated Q → balanced trust between model dynamics and data  

---

## Diagnostic Strategy

Rather than using likelihood, this script relies on **innovation diagnostics**, which are more interpretable at the calibration stage.

### 1. Innovation Whitening

For each time step, the innovation is whitened:

zₜ = Sₜ⁻¹ᐟ² dₜ

where Sₜ is the innovation covariance.

If Q is correctly scaled:
- zₜ behaves approximately like a standard normal vector
- The expected norm satisfies E‖z‖ ≈ √n, where n is the state dimension

### 2. State Covariance Stability

The trace of the posterior state covariance Pₜ is monitored:
- Collapse of Pₜ → Q too small
- Explosive growth of Pₜ → Q too large
- Stable Pₜ → acceptable regime

---

## Implementation Details

- Q is anchored to data using the covariance of daily yield changes
- A small grid of scaling factors is tested
- No likelihood, α tuning, or rolling covariances are used
- The script is intentionally minimal and diagnostic-only

---

## Typical Interpretation of Output

Three regimes typically appear:

- **Q too small**  
  Whitening norms far exceed √n, and P collapses

- **Balanced Q**  
  Whitening norms are close to √n, and P remains stable

- **Q too large**  
  Whitening norms fall below √n, and P grows excessively

The goal is to identify a **stable operating range**, not a precise optimum.

---

## Recommended Usage

1. Run the script
2. Identify the range of Q scales producing near-whitened innovations
3. Select a representative Q within that range
4. Fix Q for all subsequent likelihood-based R comparisons

---

## Why Likelihood Is Not Used Here

Likelihood is a **model comparison tool**, not a calibration tool.

At this stage:
- Likelihood is highly sensitive to Q mis-scaling
- Innovation diagnostics provide clearer, more robust guidance

Likelihood evaluation should only be performed *after* Q has been calibrated.

---

## References

- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*
- Jazwinski, A. H. (1970). *Stochastic Processes and Filtering Theory*
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of observation, background and analysis-error statistics*

---

## Summary

This script establishes a principled and interpretable foundation for process-noise calibration. By isolating Q diagnostics from likelihood evaluation, it ensures that subsequent comparisons of observation error models are statistically meaningful and not driven by scale artifacts.
