# Preregistration: Quantum Feedback Memory Space

Author: Máté Róbert (ORCID: 0009-0003-3986-6039)  
Repository: https://github.com/silentnoisehun/quantum-feedback-memory-space  
Date of Registration: 2026-11-07 10:45:00 UTC  
Script: `scripts/feedback_dual_test.py`, seeds `[20261101, 20261102, 20261103, 20261104, 20261105, 20261106]`

## 1. Physical Question and Hypothesis

Does a closed feedback loop between a numerical 2D Gross-Pitaevskii Equation (GPE) fluid simulation on a host CPU and physical transmon qubits on an IBM Quantum Processor (QPU) retain more state memory, reduce modal entropy, or exhibit non-classical coherence compared to an identically noise-matched textbook classical model?

- **H0 (Standard Physics):** The QPU communicates with the host CPU purely via classical numbers sent over the internet API. Subject to physical readout error ($P(1|0) \approx 0.012$, $P(0|1) \approx 0.018$) and finite Poisson/binomial shot noise ($N = 1024$), the closed loop behaves identically to textbook density matrix evolution. Modal entropy difference $\Delta S = S_{\text{QPU}} - S_{\text{Model}}$ is zero within sampling error ($p \ge 0.01$).
- **H1 ("Memory in Space"):** The transmon physical environment imparts an anomalous feedback bias or state retention that systematically reduces modal entropy in the fluid wavefunction ($p < 0.01$, $\Delta S < -0.05$).

## 2. Experimental Architecture

1. **Fluid Engine:** Split-step Fourier 2D GPE solver on polar-like cylindrical mesh ($N_r = 64, N_\theta = 64, \Delta t = 0.002, g = 50.0, \Omega = 1.0$).
2. **Modal Extraction:** Extracts modal energies across 16 angular harmonics $m \in [0..15]$: dominant intensity $A$, centroid mode frequency $f$, and phase $\phi$.
3. **Encoding:** Rotates 3 qubits with angles $\theta_0 = A$, $\theta_1 = 2\pi(f/15)$, $\theta_2 = \phi \pi$.
4. **Decoding:** Unbiased inversion of marginal measurement probabilities $p_0, p_1, p_2$.
5. **Feedback:** Modulates the fluid wavefunction with non-linear excitation operator $\text{excite}(A_q, f_q, \phi_q)$.

## 3. Decision Rule

- Significance threshold: $\alpha = 0.01$ (paired t-test and Wilcoxon signed-rank test across 6 independent seeds).
- Direction: One-sided lower tail for entropy ($\Delta S < -0.05$).
