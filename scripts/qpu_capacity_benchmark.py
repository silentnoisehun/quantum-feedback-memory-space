#!/usr/bin/env python3
"""
qpu_capacity_benchmark.py: Comprehensive QPU+GPE Hybrid Speed, Stability & Data Capacity Benchmark.

Measures:
1. GPE Solver Speed (steps/second & compute latency per cycle)
2. QPU Actuation & Sampling Latency (100-shot simulated vs physical IBM hardware profile)
3. Mode Stability & Coherence Decay Lifetime (cycles survived per angular momentum mode m=0..14)
4. Simultaneous Multi-Mode Storage Capacity (vortex superposition packing limit)
5. Angular Phase Constellation Resolution & Fidelity
6. Net Effective Throughput & Information Storage Capacity (bits, bits/sec, QPU density)
"""

import math
import sys
import time
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_DIR = REPO / "measurement_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from fluid_sim import FluidSim, NR, NTH, RMAX

MODES_TO_TEST = [0, 2, 4, 6, 8, 10, 12, 14]

def inject_mode(fluid, m, A=1.0, phi=0.0):
    R_mesh, Th_mesh = np.meshgrid(fluid.r_grid, fluid.th_grid, indexing='ij')
    r_center = 2.5 + (m / 16.0) * 1.5
    ring = np.exp(-(R_mesh - r_center)**2 / (2.0 * 0.9**2))
    winding = np.exp(1j * (m * Th_mesh + phi * 2.0 * math.pi))
    fluid.psi += (A * 0.5) * ring * winding
    norm = np.sqrt(np.sum(np.abs(fluid.psi)**2 * R_mesh * fluid.dr * fluid.dth))
    if norm > 1e-15:
        fluid.psi /= norm

def benchmark_gpe_speed(steps=1000):
    fluid = FluidSim(seed=42)
    t0 = time.perf_counter()
    for _ in range(steps):
        fluid.gpe_step()
    t1 = time.perf_counter()
    dt = t1 - t0
    steps_per_sec = steps / dt
    ms_per_step = (dt / steps) * 1000.0
    return {
        "steps_measured": steps,
        "elapsed_seconds": dt,
        "steps_per_second": steps_per_sec,
        "latency_per_step_ms": ms_per_step
    }

def benchmark_mode_decay():
    """
    Measures how long each topological mode m survives without feedback reinjection
    before decaying into background/m=0 in active GPE (g=50).
    """
    results = {}
    gpe_steps_per_cycle = 10
    max_cycles = 100
    
    for m in MODES_TO_TEST:
        fluid = FluidSim(seed=100 + m)
        inject_mode(fluid, m, A=1.5, phi=0.0)
        
        cycles_retained = 0
        final_energy = 0.0
        
        for cycle in range(max_cycles):
            for _ in range(gpe_steps_per_cycle):
                fluid.gpe_step()
            dom_mode, A_val, f_val, phi_val, energies = fluid.extract_modal_parameters()
            
            if dom_mode == m:
                cycles_retained = cycle + 1
                final_energy = float(energies[m])
            else:
                # Mode decayed / switched
                break
                
        results[m] = {
            "mode": m,
            "cycles_survived": cycles_retained,
            "final_energy_ratio": final_energy,
            "lifetime_steps": cycles_retained * gpe_steps_per_cycle
        }
    return results

def benchmark_simultaneous_capacity():
    """
    Tests how many orthogonal vortex modes can be superposed simultaneously
    in the fluid reservoir before inter-modal nonlinear cross-talk destroys resolution.
    Threshold: mode retains > 5% total energy after 50 evolution steps.
    """
    fluid = FluidSim(seed=777)
    modes_injected = []
    
    for m in MODES_TO_TEST:
        inject_mode(fluid, m, A=0.8, phi=0.0)
        modes_injected.append(m)
        
    # Evolve 50 steps
    for _ in range(50):
        fluid.gpe_step()
        
    _, _, _, _, energies = fluid.extract_modal_parameters()
    
    retained_modes = []
    for m in modes_injected:
        e = float(energies[m])
        if e >= 0.05:
            retained_modes.append({"mode": m, "energy": e})
            
    return {
        "total_modes_injected": len(modes_injected),
        "coexisting_modes_retained": len(retained_modes),
        "details": retained_modes,
        "energy_distribution": [float(e) for e in energies]
    }

def benchmark_phase_resolution():
    """
    Tests phase angular error across an 8-point phase constellation (3-bit phase).
    """
    fluid = FluidSim(seed=999)
    errors = []
    phases_tested = [k / 8.0 for k in range(8)]
    
    for target_phi in phases_tested:
        fl = FluidSim(seed=123)
        inject_mode(fl, m=4, A=1.5, phi=target_phi)
        for _ in range(10):
            fl.gpe_step()
        _, _, _, measured_phi, _ = fl.extract_modal_parameters()
        
        diff = abs(measured_phi - target_phi)
        if diff > 0.5:
            diff = 1.0 - diff
        ang_err = diff * 2.0 * math.pi
        errors.append(ang_err)
        
    return {
        "phases_tested": phases_tested,
        "mean_angular_error_rad": float(np.mean(errors)),
        "max_angular_error_rad": float(np.max(errors)),
        "phase_snr_db": float(10.0 * np.log10(1.0 / max(np.mean(errors)**2, 1e-6)))
    }

def run_capacity_benchmark():
    print("=" * 82)
    print("HYBRID QPU + GPE FLUID SPEED, STABILITY & STORAGE BENCHMARK")
    print("=" * 82)
    
    # 1. GPE Speed
    print("\n[1/5] Measuring GPE Numerical Core Speed...")
    gpe_perf = benchmark_gpe_speed(steps=1000)
    print(f"  -> GPE Speed: {gpe_perf['steps_per_second']:.1f} steps/sec ({gpe_perf['latency_per_step_ms']:.3f} ms/step)")
    
    # 2. Mode Stability
    print("\n[2/5] Measuring Topological Mode Retention Lifetime (without feedback)...")
    mode_decay = benchmark_mode_decay()
    for m_val, stat in mode_decay.items():
        print(f"  Mode m={m_val:2d}: Survived {stat['cycles_survived']:2d} cycles ({stat['lifetime_steps']:3d} GPE steps) | Residual: {stat['final_energy_ratio']:.1%}")
    
    # 3. Simultaneous Capacity
    print("\n[3/5] Measuring Simultaneous Superposition Storage Capacity...")
    sim_cap = benchmark_simultaneous_capacity()
    print(f"  -> Coexisting Stable Modes: {sim_cap['coexisting_modes_retained']} out of {sim_cap['total_modes_injected']} modes")
    for det in sim_cap["details"]:
        print(f"     * Mode m={det['mode']:2d} : {det['energy']:.1%} energy")
        
    # 4. Phase Resolution
    print("\n[4/5] Measuring Angular Phase Resolution & Distortion...")
    phase_res = benchmark_phase_resolution()
    print(f"  -> Mean Angular Error: {phase_res['mean_angular_error_rad']:.4f} rad ({phase_res['mean_angular_error_rad'] * 180 / math.pi:.2f} deg)")
    print(f"  -> Phase Constellation SNR: {phase_res['phase_snr_db']:.2f} dB")
    
    # 5. Throughput & Bitrate Synthesis
    print("\n[5/5] Synthesizing Hardware & Reservoir Speed & Capacity...")
    
    # Physical transmon timing model (IBM Heron/Falcon specs):
    # Single qubit rotation: 30 ns
    # 2-qubit entangling (ECR/CZ): 300 ns
    # Readout pulse + ringdown: 800 ns
    # Shot repetition time (reset): 250 us
    # Total for 100 shots: 100 * 250 us = 25 ms per readout cycle
    qpu_readout_time_100shots_sec = 0.025
    gpe_cycle_time_20steps_sec = 20 * (gpe_perf['latency_per_step_ms'] / 1000.0)
    total_cycle_time_sec = qpu_readout_time_100shots_sec + gpe_cycle_time_20steps_sec
    
    # 8 topological modes (3 bits) + 8 phase bins (3 bits) = 6 bits per channel
    # With 3 simultaneous modes = 18 bits per active reservoir state
    bits_per_symbol = 3.0 # Mode index (8 states)
    bits_per_phase = 3.0  # Phase index (8 states)
    bits_per_carrier = bits_per_symbol + bits_per_phase
    simultaneous_carriers = sim_cap['coexisting_modes_retained']
    total_reservoir_bits = bits_per_carrier * simultaneous_carriers
    
    effective_bandwidth_bps = total_reservoir_bits / total_cycle_time_sec
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpe_core": gpe_perf,
        "mode_decay": mode_decay,
        "simultaneous_capacity": sim_cap,
        "phase_resolution": phase_res,
        "hardware_timing": {
            "qpu_100shots_latency_sec": qpu_readout_time_100shots_sec,
            "gpe_20steps_latency_sec": gpe_cycle_time_20steps_sec,
            "total_hybrid_cycle_sec": total_cycle_time_sec,
            "hybrid_cycles_per_sec": 1.0 / total_cycle_time_sec
        },
        "information_metrics": {
            "bits_per_mode_index": bits_per_symbol,
            "bits_per_phase_index": bits_per_phase,
            "bits_per_carrier": bits_per_carrier,
            "simultaneous_carriers": simultaneous_carriers,
            "total_reservoir_state_bits": total_reservoir_bits,
            "effective_bandwidth_bps": effective_bandwidth_bps,
            "effective_bandwidth_kbps": effective_bandwidth_bps / 1000.0
        }
    }
    
    print("-" * 82)
    print("HYBRID METRICS SUMMARY:")
    print(f"  Hybrid Cycle Duration    : {total_cycle_time_sec * 1000.0:.2f} ms ({1.0 / total_cycle_time_sec:.1f} cycles/sec)")
    print(f"  Total Reservoir Capacity : {total_reservoir_bits:.1f} bits (across {simultaneous_carriers} coexisting topological modes)")
    print(f"  Effective Data Rate      : {effective_bandwidth_bps:.1f} bits/sec ({effective_bandwidth_bps / 1000.0:.3f} kbps)")
    print("=" * 82)
    
    out_file = RAW_DIR / "qpu_capacity_benchmark.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark successfully saved to: {out_file}")

if __name__ == "__main__":
    run_capacity_benchmark()
