# Quantum Feedback Memory Space

Dual-test evaluation of a closed feedback loop between a CPU Gross-Pitaevskii (GPE) fluid simulation and transmon quantum hardware.

**Author:** Máté Róbert (ORCID: [0009-0003-3986-6039](https://orcid.org/0009-0003-3986-6039))  
**Preregistration:** `docs/PREREGISTRATION-feedback_memory.md`

---

## The Question

Does feeding quantum measurement outcomes back into a classical hydrodynamic simulation retain more quantum coherence or state memory than an identically noise-matched classical control model?

The GPE fluid simulation (`FluidSim`) runs locally in NumPy on the host CPU. It interacts with the QPU solely via numerical values exchanged over the network API. This benchmark directly evaluates whether the hardware loop introduces any anomalous memory effect or if its dynamics are fully explained by textbook quantum mechanics + hardware readout noise.

---

## Experimental Design

The dual-test compares three arms over 40 iterations across 6 independent pseudorandom seeds (`20261101`..`20261106`):

1. **Arm I (Ideal Continuum):** Deterministic noiseless feedback reference.
2. **Arm Q (Hardware-Matched QPU):** Full transmon error profile ($T_1$ relaxation, thermal ground-state excitation, binomial shot noise with $N=1024$).
3. **Arm M (Noise-Matched Classical Control):** Exact textbook density matrix model with calibrated identical noise levels and zero-offset at null.

```
       [ FluidSim (CPU) ]
           │          ▲
  (A, f, φ)│          │ (A_q, f_q, φ_q)
           ▼          │
      [ Encode ]   [ Decode ]
           │          ▲
           ▼          │
       [ Transmon QPU / Model ]
```

---

## Results

Evaluated across 6 independent seeds (stationary iterations 10..40):

| Seed | Ideal Entropy | QPU Arm Entropy | Model Arm Entropy | Difference (Q - M) | Mode Flips (Q / M) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `20261101` | 0.9985 | 0.9747 | 1.0031 | -0.0285 | 2 / 2 |
| `20261102` | 0.9975 | 1.0048 | 0.9896 | +0.0152 | 2 / 2 |
| `20261103` | 0.9981 | 1.0153 | 0.9689 | +0.0464 | 2 / 2 |
| `20261104` | 0.9980 | 1.0083 | 0.9936 | +0.0147 | 2 / 2 |
| `20261105` | 0.9977 | 0.9963 | 0.9720 | +0.0243 | 2 / 2 |
| `20261106` | 0.9974 | 0.9984 | 0.9947 | +0.0037 | 2 / 2 |

- **Mean Entropy Difference ($\Delta S$):** $+0.0126 \pm 0.025$
- **Paired t-test:** $t = 1.25$, $p = 0.2657$
- **Wilcoxon signed-rank test:** $p = 0.3125$
- **Mode Flips / State Retention:** Identical across all runs ($2$ flips per arm per seed).

### Verdict

$$\mathbf{H_0\ CONFIRMED}$$

The closed-loop dynamics between the QPU and the fluid simulation are statistically indistinguishable from textbook quantum mechanics with standard transmon readout noise ($p = 0.266$). The QPU adds no anomalous state retention or non-classical coherence to the classical fluid.

---

## Reproduction

```bash
python scripts/feedback_dual_test.py
```
Output data saved in `measurement_raw/feedback_dual_test_results.json`.
