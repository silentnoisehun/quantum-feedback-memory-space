# Preregistration: Scaled 100-Shot Topological Attractor Memory & Red-Team Audit

Author: Máté Róbert (ORCID: 0009-0003-3986-6039)  
Repository: https://github.com/silentnoisehun/quantum-feedback-memory-space  
Date of Registration: 2026-11-07 11:15:00 UTC  
Script: `scripts/scaled_topological_memory.py`  
Seeds: `[20261110, 20261111, 20261112, 20261113, 20261114, 20261115]`

## 1. Core Hypothesis

A hybrid closed-loop architecture coupling a non-linear 2D Gross-Pitaevskii Equation (GPE) fluid simulation to a transmon quantum processor can store, stabilize, and reconstruct multi-bit discrete symbols using only $N = 100$ measurement shots per feedback iteration.

The non-linear self-interaction term ($g|\psi|^2\psi$) acts as an active topological filter that contracts the binomial measurement noise ($\sigma \approx 1/(2\sqrt{N}) = 5\%$) into discrete quantized angular momentum attractor basins ($m \in \mathbb{Z}$), achieving:
1. **Bit Error Rate (BER):** $\text{BER} < 0.01$ (symbol reconstruction fidelity $> 99.0\%$).
2. **Topological Protection:** Stability against readout confusion noise ($P(1|0) = 0.012$, $P(0|1) = 0.018$) across $\ge 25$ feedback cycles.

## 2. Experimental Protocol

- **Discrete Symbol Constellation:** 8 discrete attractor states ($3\text{ bits}$ per cycle):
  - $m \in \{0, 2, 4, 6, 8, 10, 12, 14\}$
- **Feedback Sampling:** Exactly $N = 100$ physical shots per iteration.
- **Leakage Elimination (Clean Slate):** The classical feedback channel passes *only* the decoded numeric tuple $(A_q, m_q, \phi_q)$ across iterations; no hidden state or NumPy buffer persists across the quantum boundary.

## 3. Red-Team Controls (Adversarial Null Models)

To rule out trivial artifacts, every benchmark run must simultaneously evaluate two adversarial controls:
1. **Control A (Linear Dispersion Null, $g = 0$):** Non-linearity turned off. Without $g|\psi|^2$, higher harmonics must disperse and fail to lock.
2. **Control B (Shuffled Phase Null):** Quantum feedback angle is phase-scrambled uniformly in $[0, 2\pi)$. Must show zero retention.

## 4. Decision Rule

- **PASS:** In the active GPE loop ($g = 50.0$), mean reconstruction fidelity $> 95.0\%$, symbol retention $> 90.0\%$, and both adversarial null controls fail ($< 25\%$ retention).
- **FAIL:** Fidelity $\le 95.0\%$ or if the linear null ($g=0$) also retains the symbols (indicating trivial persistence rather than non-linear attractor dynamics).
