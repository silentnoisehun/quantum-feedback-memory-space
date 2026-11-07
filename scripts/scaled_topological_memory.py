#!/usr/bin/env python3
"""
scaled_topological_memory.py: Scaled 100-Shot Topological Memory Engine & Red-Team Audit.
Preregistration: docs/PREREGISTRATION-scaled_100shot_memory.md

Tests an 8-symbol (3-bit) topological constellation using exactly N = 100 shots per cycle.
Includes full Red-Team adversarial controls:
  1. Active Non-linear Attractor Loop (g = 50.0)
  2. Linear Dispersion Control (g = 0.0) -> MUST FAIL (proving non-linearity is required)
  3. Phase-Scrambled Control -> MUST FAIL (proving quantum phase coherence is required)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_DIR = REPO / "measurement_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from fluid_sim import FluidSim, NR, NTH, RMAX

# 8 discrete symbols (3 bits): m in {0, 2, 4, 6, 8, 10, 12, 14}
SYMBOLS_8 = {i: {"m": i * 2, "phi": (i / 8.0)} for i in range(8)}

P_0_GIVEN_1 = 0.018
P_1_GIVEN_0 = 0.012

def inject_mode(fluid, m, A=1.0, phi=0.0):
    R_mesh, Th_mesh = np.meshgrid(fluid.r_grid, fluid.th_grid, indexing='ij')
    r_center = 2.5 + (m / 16.0) * 1.5
    ring = np.exp(-(R_mesh - r_center)**2 / (2.0 * 0.9**2))
    winding = np.exp(1j * (m * Th_mesh + phi * 2.0 * math.pi))
    fluid.psi += (A * 0.5) * ring * winding
    norm = np.sqrt(np.sum(np.abs(fluid.psi)**2 * R_mesh * fluid.dr * fluid.dth))
    if norm > 1e-15:
        fluid.psi /= norm

def run_loop(symbol_id, condition="active_gpe", shots=100, cycles=20, gpe_steps=20, seed=20261110):
    sym = SYMBOLS_8[symbol_id]
    fluid = FluidSim(seed=seed)
    rng = np.random.default_rng(seed + symbol_id * 50)
    
    # Red-Team: In linear condition, disable GPE non-linearity (g = 0)
    if condition == "linear_null":
        # Overwrite G to 0 in potential phase
        pass
        
    inject_mode(fluid, sym["m"], A=1.5, phi=sym["phi"])
    
    cycle_history = []
    
    for cycle in range(cycles):
        for _ in range(gpe_steps):
            if condition == "linear_null":
                # Linear kinetic + harmonic trap only (no g|psi|^2)
                density = np.abs(fluid.psi)**2
                phase_v = -0.5 * fluid.potential * 0.002 * 0.5
                fluid.psi *= np.exp(1j * phase_v)
                psi_k = np.fft.fft2(fluid.psi)
                psi_k *= fluid.kinetic_prop
                fluid.psi = np.fft.ifft2(psi_k)
                fluid.psi *= np.exp(1j * phase_v)
            else:
                fluid.gpe_step()
                
        mode_argmax, a_fluid, f_fluid, phi_fluid, energies = fluid.extract_modal_parameters()
        
        # Quantum encoding
        theta0 = a_fluid
        theta1 = 2.0 * math.pi * (mode_argmax / 16.0)
        theta2 = phi_fluid * math.pi
        
        p0 = math.sin(theta0 / 2.0)**2
        p1 = math.sin(theta1 / 2.0)**2
        p2 = math.sin(theta2 / 2.0)**2
        
        # Readout noise
        p0_n = p0 * (1.0 - P_0_GIVEN_1) + (1.0 - p0) * P_1_GIVEN_0
        p1_n = p1 * (1.0 - P_0_GIVEN_1) + (1.0 - p1) * P_1_GIVEN_0
        p2_n = p2 * (1.0 - P_0_GIVEN_1) + (1.0 - p2) * P_1_GIVEN_0
        
        # 100-shot sampling
        k0 = rng.binomial(shots, min(max(p0_n, 0.0), 1.0))
        k1 = rng.binomial(shots, min(max(p1_n, 0.0), 1.0))
        k2 = rng.binomial(shots, min(max(p2_n, 0.0), 1.0))
        
        p_m0 = k0 / shots
        p_m1 = k1 / shots
        p_m2 = k2 / shots
        
        denom = max(1.0 - P_0_GIVEN_1 - P_1_GIVEN_0, 0.01)
        p_m0 = min(max((p_m0 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        p_m1 = min(max((p_m1 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        p_m2 = min(max((p_m2 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        
        a_q = 2.0 * math.asin(math.sqrt(p_m0))
        m_cont = math.asin(math.sqrt(p_m1)) / math.pi * 16.0
        # Quantize to nearest 2-step basin: {0, 2, 4, 6, 8, 10, 12, 14}
        m_est = int(round(m_cont / 2.0) * 2)
        m_est = min(max(m_est, 0), 14)
        
        if condition == "phase_scrambled":
            phi_q = rng.uniform(0.0, 1.0)  # Destroy phase coherence
        else:
            phi_q = 2.0 * math.asin(math.sqrt(p_m2)) / math.pi
            
        # Perturbative feedback only — excite() uses amp = A * 0.1
        f_est = m_est  # frequency index = mode number
        fluid.excite(a_q, f_est, phi_q)
        
        retained = (mode_argmax == sym["m"])
        fidelity = float(energies[sym["m"]])
        
        cycle_history.append({
            "cycle": cycle,
            "measured_mode": mode_argmax,
            "target_mode": sym["m"],
            "retained": retained,
            "fidelity": fidelity
        })
        
    return cycle_history

def run_deep_audit():
    print("=" * 82)
    print("RED-TEAM DEEP AUDIT: SCALED 100-SHOT TOPOLOGICAL QUANTUM MEMORY")
    print("Testing 8 Discrete Symbols (3 Bits/Iteration) Under Adversarial Controls")
    print("=" * 82)
    
    conditions = ["active_gpe", "linear_null", "phase_scrambled"]
    audit_report = {}
    
    for cond in conditions:
        print(f"\n---> Evaluating Condition: [{cond.upper()}] ...")
        sym_scores = []
        fid_scores = []
        
        for sym_id in range(8):
            target_m = SYMBOLS_8[sym_id]["m"]
            hist = run_loop(sym_id, condition=cond, shots=100, cycles=20, gpe_steps=20)
            ret = np.mean([h["retained"] for h in hist])
            fid = np.mean([h["fidelity"] for h in hist])
            final_m = hist[-1]["measured_mode"]
            success = (final_m == target_m)
            
            sym_scores.append(ret)
            fid_scores.append(fid)
            print(f"  Symbol #{sym_id} (m={target_m:2d}): Final m={final_m:2d} | "
                  f"Retention={ret:6.1%} | Mean Fidelity={fid:6.1%} | {'[LOCK]' if success else '[FAIL]'}")
                  
        mean_ret = float(np.mean(sym_scores))
        mean_fid = float(np.mean(fid_scores))
        print(f"  ==> CONDITION SUMMARY [{cond}]: Mean Retention={mean_ret:6.1%} | Mean Fidelity={mean_fid:6.1%}")
        audit_report[cond] = {"mean_retention": mean_ret, "mean_fidelity": mean_fid}
        
    print("\n" + "=" * 82)
    print("RED-TEAM ARBITRATION VERDICT:")
    gpe_ret = audit_report["active_gpe"]["mean_retention"]
    lin_ret = audit_report["linear_null"]["mean_retention"]
    scr_ret = audit_report["phase_scrambled"]["mean_retention"]
    
    print(f"  1. Active GPE Non-linear Memory Retention: {gpe_ret:.1%}")
    print(f"  2. Linear Dispersion Null Retention (g=0) : {lin_ret:.1%}")
    print(f"  3. Phase-Scrambled Null Retention         : {scr_ret:.1%}")
    
    is_valid = (gpe_ret > 0.85) and (lin_ret < 0.35) and (scr_ret < 0.35)
    if is_valid:
        verdict = "RED-TEAM AUDIT PASSED: Memory is strictly non-linear and phase-coherent."
    else:
        verdict = "RED-TEAM AUDIT FAILED."
    print(f"  VERDICT: {verdict}")
    print("=" * 82)
    
    out_file = RAW_DIR / "scaled_100shot_red_team_audit.json"
    out_file.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(f"Audit log saved to: {out_file}")

if __name__ == "__main__":
    run_deep_audit()
