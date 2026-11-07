"""
GPE Fluid Simulation Engine (Python Reference + C-compatible pure NumPy implementation)
Used by the QPU loop runner to ensure identical physics to the Rust SIMD solver.
"""

import math
import numpy as np

NR = 64
NTH = 64
RMAX = 8.0
DT = 0.002
G = 50.0
OMEGA = 1.0
F_MODE_INDEX_MAX = 15.0

class FluidSim:
    def __init__(self, seed=20261026):
        self.dr = RMAX / NR
        self.dth = 2.0 * math.pi / NTH
        self.r_grid = (np.arange(NR) + 0.5) * self.dr
        self.th_grid = np.arange(NTH) * self.dth

        # Potential V(r) = omega^2 * r^2
        R_mesh, _ = np.meshgrid(self.r_grid, self.th_grid, indexing='ij')
        self.potential = OMEGA * OMEGA * (R_mesh ** 2)

        # Kinetic propagator in 2D Fourier space
        kr = np.fft.fftfreq(NR, d=self.dr) * 2.0 * math.pi
        kth = np.fft.fftfreq(NTH, d=self.dth) * 2.0 * math.pi
        Kr, Kth = np.meshgrid(kr, kth, indexing='ij')
        k2 = Kr**2 + Kth**2
        self.kinetic_prop = np.exp(-1j * 0.5 * k2 * DT)

        # Initial ground state Gaussian + perturbation
        rng = np.random.default_rng(seed)
        gauss = np.exp(-R_mesh**2 / 4.0)
        pert = 0.01 * np.exp(1j * rng.uniform(0, 2*math.pi, size=(NR, NTH)))
        self.psi = (gauss + pert * gauss).astype(np.complex128)

        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.psi)**2 * R_mesh * self.dr * self.dth))
        if norm > 1e-15:
            self.psi /= norm

    def gpe_step(self):
        # Half step potential
        density = np.abs(self.psi)**2
        phase_v = -0.5 * DT * 0.5 * (self.potential + G * density)
        self.psi *= np.exp(1j * phase_v)

        # Kinetic step (2D FFT)
        psi_k = np.fft.fft2(self.psi)
        psi_k *= self.kinetic_prop
        self.psi = np.fft.ifft2(psi_k)

        # Half step potential
        density = np.abs(self.psi)**2
        phase_v = -0.5 * DT * 0.5 * (self.potential + G * density)
        self.psi *= np.exp(1j * phase_v)

    def excite(self, A, f, phi):
        """Mathematically corrected excitation operator."""
        target_rad = (f / F_MODE_INDEX_MAX) * RMAX
        target_th = phi * 2.0 * math.pi

        # Mode centers
        centers = []
        for m in range(16):
            r = (m / 15.0) * RMAX
            th = 2.0 * math.pi * m / 16.0
            centers.append((r, th))

        best_mode = 0
        best_dist = 1e9
        for m, (cr, cth) in enumerate(centers):
            d_rad = target_rad - cr
            d_ang = abs(target_th - cth)
            if d_ang > math.pi:
                d_ang = 2.0 * math.pi - d_ang
            dist = math.hypot(d_rad, d_ang)
            if dist < best_dist:
                best_dist = dist
                best_mode = m

        center_r, center_th = centers[best_mode]
        sigma = RMAX * 0.15
        amp = A * 0.1

        R_mesh, Th_mesh = np.meshgrid(self.r_grid, self.th_grid, indexing='ij')
        d_r = R_mesh - center_r
        d_th = np.abs(Th_mesh - center_th)
        d_th = np.minimum(d_th, 2.0 * math.pi - d_th)
        dist2 = d_r**2 + d_th**2

        exc = amp * np.exp(-dist2 / (2.0 * sigma**2))
        phase = phi * 2.0 * math.pi
        self.psi += exc * np.exp(1j * phase)

    def extract_modal_parameters(self):
        """Extract modal energies and compute (A, f, phi)."""
        R_mesh, Th_mesh = np.meshgrid(self.r_grid, self.th_grid, indexing='ij')
        energies = np.zeros(16)
        for m in range(16):
            basis = np.exp(-1j * m * Th_mesh)
            proj = np.sum(self.psi * basis * R_mesh * self.dr * self.dth)
            energies[m] = np.abs(proj)**2

        total = np.sum(energies)
        if total > 1e-15:
            energies /= total

        argmax = int(np.argmax(energies))
        # Intensity A: energy in dominant mode
        A = float(np.max(energies))
        # Frequency f: centroid mode index
        f = float(np.sum(np.arange(16) * energies))
        # Phase phi: phase of dominant projection
        dominant_basis = np.exp(-1j * argmax * Th_mesh)
        proj_dom = np.sum(self.psi * dominant_basis * R_mesh * self.dr * self.dth)
        phi = float((np.angle(proj_dom) % (2.0 * math.pi)) / (2.0 * math.pi))

        return argmax, A, f, phi, energies
