#!/usr/bin/env python3
"""
low_shot_vortex_memory.py: 20-Shot Topological Memory & State Reconstruction Engine.
Preregistration: docs/PREREGISTRATION-feedback_memory.md

Demonstrates low-energy, low-shot (N = 20 shots) information encoding, retention,
and reconstruction in a closed-loop hybrid Gross-Pitaevskii (GPE) + Transmon architecture.

Uses topologically quantized angular harmonics: psi_exc ~ exp(i * m * theta)
Tests 4 distinct discrete memory symbols:
  Symbol 0: Fundamental ground state (m = 0)
  Symbol 1: Vortex triplet (m = 3)
  Symbol 2: Vortex septet (m = 7)
  Symbol 3: High-harmonic cluster (m = 11)
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

from fluid_sim import FluidSim

SYMBOLS = {
    0: {"name": "Attractor_Mode_0", "m": 0, "phi": 0.0},
    1: {"name": "Attractor_Mode_4", "m": 4, "phi": 0.25},
    2: {"name": "Attractor_Mode_8", "m": 8, "phi": 0.50},
    3: {"name": "Attractor_Mode_12", "m": 12, "phi": 0.75},
}

P_0_GIVEN_1 = 0.018
P_1_GIVEN_0 = 0.012

def inject_topological_mode(fluid, m, A=1.0, phi=0.0):
    """Injects a topologically quantized winding state exp(i * m * theta)."""
    R_mesh, Th_mesh = np.meshgrid(fluid.r_grid, fluid.th_grid, indexing='ij')
    # Radial ring centered at r = 3.0
    r_center = 2.5 + (m / 15.0) * 1.5
    sigma = 0.9
    ring = np.exp(-(R_mesh - r_center)**2 / (2.0 * sigma**2))
    winding = np.exp(1j * (m * Th_mesh + phi * 2.0 * math.pi))
    fluid.psi += (A * 0.5) * ring * winding
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(fluid.psi)**2 * R_mesh * fluid.dr * fluid.dth))
    if norm > 1e-15:
        fluid.psi /= norm

def run_symbol_memory_loop(symbol_id, shots=20, cycles=30, gpe_steps=20, seed=20261107):
    sym = SYMBOLS[symbol_id]
    fluid = FluidSim(seed=seed)
    rng = np.random.default_rng(seed + symbol_id * 100)
    
    # Initial write: inject target mode
    inject_topological_mode(fluid, sym["m"], A=1.5, phi=sym["phi"])
    
    cycle_history = []
    
    for cycle in range(cycles):
        # 1. Non-linear GPE evolution
        for _ in range(gpe_steps):
            fluid.gpe_step()
            
        # 2. Extract modal spectrum
        mode_argmax, a_fluid, f_fluid, phi_fluid, energies = fluid.extract_modal_parameters()
        
        # 3. Quantum Encoding
        # We encode the dominant detected mode directly: theta1 = 2pi * (mode_argmax / 15)
        # and phase: theta2 = phi_fluid * pi
        theta0 = a_fluid
        theta1 = 2.0 * math.pi * (mode_argmax / 15.0)
        theta2 = phi_fluid * math.pi
        
        p0 = math.sin(theta0 / 2.0)**2
        p1 = math.sin(theta1 / 2.0)**2
        p2 = math.sin(theta2 / 2.0)**2
        
        # Hardware readout assignment noise
        p0_noisy = p0 * (1.0 - P_0_GIVEN_1) + (1.0 - p0) * P_1_GIVEN_0
        p1_noisy = p1 * (1.0 - P_0_GIVEN_1) + (1.0 - p1) * P_1_GIVEN_0
        p2_noisy = p2 * (1.0 - P_0_GIVEN_1) + (1.0 - p2) * P_1_GIVEN_0
        
        # 20-shot binomial sampling (extreme shot noise!)
        k0 = rng.binomial(shots, min(max(p0_noisy, 0.0), 1.0))
        k1 = rng.binomial(shots, min(max(p1_noisy, 0.0), 1.0))
        k2 = rng.binomial(shots, min(max(p2_noisy, 0.0), 1.0))
        
        p_m0 = k0 / shots
        p_m1 = k1 / shots
        p_m2 = k2 / shots
        
        # Readout inversion
        denom = max(1.0 - P_0_GIVEN_1 - P_1_GIVEN_0, 0.01)
        p_m0 = min(max((p_m0 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        p_m1 = min(max((p_m1 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        p_m2 = min(max((p_m2 - P_1_GIVEN_0) / denom, 0.0), 1.0)
        
        # Decoding with topological attractor basin projection (error-correcting threshold)
        a_q = 2.0 * math.asin(math.sqrt(p_m0))
        m_cont = math.asin(math.sqrt(p_m1)) / math.pi * 15.0
        m_est = int(round(m_cont / 4.0) * 4)
        m_est = min(max(m_est, 0), 12)
        phi_q = 2.0 * math.asin(math.sqrt(p_m2)) / math.pi
        
        # 4. FEEDBACK: Topological Mode Reinforcement
        inject_topological_mode(fluid, m_est, A=a_q, phi=phi_q)
        
        retained = (mode_argmax == sym["m"])
        fidelity = float(energies[sym["m"]])
        
        cycle_history.append({
            "cycle": cycle,
            "measured_mode": mode_argmax,
            "target_mode": sym["m"],
            "retained": retained,
            "fidelity": fidelity,
            "dominant_energy": float(np.max(energies))
        })
        
    return cycle_history

def main():
    print("=" * 80)
    print("TOPOLOGICAL QUANTUM MEMORY BENCHMARK: 20 SHOTS VS 100 SHOTS VS 1024 SHOTS")
    print("=" * 80)
    
    shot_configs = [20, 100, 1024]
    results = {}
    
    for shots in shot_configs:
        print(f"\n---> Testing Feedback Loop with N = {shots} SHOTS per cycle...")
        sym_perf = {}
        for sym_id, sym in SYMBOLS.items():
            hist = run_symbol_memory_loop(sym_id, shots=shots, cycles=20, gpe_steps=20)
            retention = np.mean([h["retained"] for h in hist])
            mean_fid = np.mean([h["fidelity"] for h in hist])
            final_mode = hist[-1]["measured_mode"]
            success = (final_mode == sym["m"])
            
            print(f"  Symbol #{sym_id} [{sym['name']:18s}]: Target m={sym['m']:2d} | "
                  f"Final m={final_mode:2d} | Retention={retention:6.1%} | "
                  f"Mean Fidelity={mean_fid:6.1%} | {'[LOCK SUCCESS]' if success else '[LOST]'}")
            
            sym_perf[sym["name"]] = {
                "m": sym["m"],
                "final_mode": final_mode,
                "retention": float(retention),
                "fidelity": float(mean_fid),
                "success": bool(success)
            }
        
        total_success = np.mean([v["success"] for v in sym_perf.values()])
        total_retention = np.mean([v["retention"] for v in sym_perf.values()])
        print(f"  --> SUMMARY (N={shots} shots): Success Rate={total_success:6.1%} | Mean Retention={total_retention:6.1%}")
        results[f"shots_{shots}"] = {"shots": shots, "success_rate": float(total_success), "retention": float(total_retention), "symbols": sym_perf}
        
    out_file = RAW_DIR / "topological_low_shot_memory_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED. Saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
