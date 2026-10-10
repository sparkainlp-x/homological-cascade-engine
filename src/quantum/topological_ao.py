"""Experimental topological filtering for multichannel optical telemetry.

The numerical output is a research prototype. It is not a validated quantum
key distribution security control and must not be used to claim attack
detection, secret-key security, or a specific latency guarantee.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.analog_calibration import AnalogSoundCalibrator
from src.deep_mechanics.nonstationary_evt import NonStationaryPOTEstimator
from src.deep_mechanics.topological_laplacian import HigherDimensionalLaplacian
from src.homological_cascade import HomologicalMembraneChain


class TopologicalAdaptiveOpticsFilter:
    """Filter sensor phase residuals and score their coexact energy.

    Parameters
    ----------
    n_subapertures : int
        Number of channels in each sensor frame. At least 160 are required by
        the configured 32/128-cell numerical 2-complex.
    sample_rate_hz : float
        Sensor sample rate used by the calibration metadata.
    coupling_j : float
        Non-negative diffusion coupling for the 1-Laplacian exponential.

    Notes
    -----
    This implementation is an experimental numerical model. The returned
    boolean is a threshold exceedance, not evidence of a physical eavesdropper.
    """

    def __init__(
        self,
        n_subapertures: int = 256,
        sample_rate_hz: float = 10000.0,
        coupling_j: float = 1.2,
    ) -> None:
        if isinstance(n_subapertures, bool) or not isinstance(n_subapertures, (int, np.integer)):
            raise TypeError("n_subapertures must be an integer")
        if n_subapertures < 160:
            raise ValueError("n_subapertures must be at least 160 for the configured 2-complex")
        coupling = float(coupling_j)
        if not np.isfinite(coupling) or coupling < 0:
            raise ValueError("coupling_j must be finite and non-negative")
        self.n_subapertures = int(n_subapertures)
        self.J = coupling

        self.calibrator = AnalogSoundCalibrator(
            sample_rate_hz=sample_rate_hz, n_channels=self.n_subapertures
        )
        self.membrane = HomologicalMembraneChain(
            n_0cells=32, n_1cells=self.n_subapertures, n_2cells=128
        )
        self.laplacian_solver = HigherDimensionalLaplacian(self.membrane.d1, self.membrane.d2)
        self.dt = 0.02
        self.U_diffusion = expm(-self.J * self.laplacian_solver.L_k * self.dt)
        self.pot_estimator = NonStationaryPOTEstimator(risk_level=1e-4, init_quantile=0.95)
        self.is_calibrated = False

    def calibrate_ambient_sky(self, ambient_optical_telemetry: np.ndarray) -> None:
        """Calibrate offsets/scales and fit a POT model from ambient telemetry.

        ``ambient_optical_telemetry`` must be shaped as
        ``(n_frames, n_subapertures)``. Calibration state is marked ready only
        after both sensor and threshold models have been fitted successfully.
        """
        telemetry = np.asarray(ambient_optical_telemetry, dtype=float)
        if telemetry.ndim != 2 or telemetry.shape[1] != self.n_subapertures:
            raise ValueError(
                f"ambient_optical_telemetry must have shape (n_frames, {self.n_subapertures})"
            )
        self.is_calibrated = False
        self.calibrator.calibrate_baseline(telemetry)
        clean_block = self.calibrator.process_block_whitened(telemetry)
        diffused_block = clean_block @ self.U_diffusion.T
        energies = np.empty(diffused_block.shape[0], dtype=float)
        for index, frame in enumerate(diffused_block):
            _, coexact, _ = self.laplacian_solver.hodge_decomposition(frame)
            energies[index] = float(np.linalg.norm(coexact) ** 2)
        self.pot_estimator.fit_initial_window(energies)
        self.is_calibrated = True

    def process_photon_frame(self, raw_optical_frame: np.ndarray) -> tuple[bool, np.ndarray, float]:
        """Filter one frame and return ``(threshold_exceeded, phase, energy)``.

        The reconstructed phase is returned as a centered residual in sensor
        units; the EVT score is computed in the calibrated whitened space.
        """
        if not self.is_calibrated:
            raise RuntimeError("Filter must be calibrated with ambient sky baseline.")
        whitened_frame = self.calibrator.process_frame(raw_optical_frame)
        filtered_whitened = self.U_diffusion @ whitened_frame
        _, coexact, _ = self.laplacian_solver.hodge_decomposition(filtered_whitened)
        energy = float(np.linalg.norm(coexact) ** 2)
        is_attack, _ = self.pot_estimator.update_dynamic_threshold(energy)
        reconstructed_phase = self.calibrator.restore_scale(filtered_whitened)
        return is_attack, reconstructed_phase, energy
