"""Topological adaptive-optics filtering and QKD anomaly triage."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.analog_calibration import AnalogSoundCalibrator
from src.deep_mechanics.nonstationary_evt import NonStationaryPOTEstimator
from src.deep_mechanics.topological_laplacian import HigherDimensionalLaplacian
from src.homological_cascade import HomologicalMembraneChain


class TopologicalAdaptiveOpticsFilter:
    """Diffuse phase turbulence and flag extreme QKD channel anomalies.

    This software filter combines a finite 2-complex, exact matrix-exponential
    diffusion over its combinatorial 1-Laplacian, robust detector calibration,
    and a non-stationary peaks-over-threshold estimator. It is a numerical
    telemetry-processing component; it does not replace physical AO hardware or
    establish cryptographic security of a QKD implementation.

    Parameters
    ----------
    n_subapertures : int, default=256
        Number of phase-sensor channels in each incoming frame.
    sample_rate_hz : float, default=10000.0
        Detector telemetry sampling rate used by the calibrator.
    coupling_j : float, default=1.2
        Non-negative diffusion coupling in the matrix exponential.
    """

    def __init__(
        self,
        n_subapertures: int = 256,
        sample_rate_hz: float = 10_000.0,
        coupling_j: float = 1.2,
    ) -> None:
        if isinstance(n_subapertures, (bool, np.bool_)) or not isinstance(
            n_subapertures, (int, np.integer)
        ):
            raise TypeError("n_subapertures must be a positive integer")
        if n_subapertures < 1:
            raise ValueError("n_subapertures must be positive")
        if isinstance(coupling_j, (bool, np.bool_)):
            raise TypeError("coupling_j must be finite and non-negative")
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
        self.laplacian_solver = HigherDimensionalLaplacian(
            self.membrane.d1, self.membrane.d2
        )

        # The fixed 20 ms diffusion horizon is the v0.6.0 specification value.
        self.diffusion_dt = 0.02
        self.U_diffusion = expm(
            -self.J * self.laplacian_solver.L_k * self.diffusion_dt
        )

        self.pot_estimator = NonStationaryPOTEstimator(
            risk_level=1e-4, init_quantile=0.95, alpha_decay=0.95
        )
        self.is_calibrated = False

    def calibrate_ambient_sky(self, ambient_optical_telemetry: np.ndarray) -> None:
        """Calibrate the detector and fit the initial ambient-energy tail model."""
        self.is_calibrated = False
        self.calibrator.calibrate_baseline(ambient_optical_telemetry)
        clean_block = self.calibrator.process_block_whitened(
            ambient_optical_telemetry
        )
        if clean_block.ndim != 2:
            raise ValueError("ambient_optical_telemetry must be a two-dimensional block")

        energies = np.empty(clean_block.shape[0], dtype=float)
        for index, frame in enumerate(clean_block):
            filtered_frame = self.U_diffusion @ frame
            _, coexact, _ = self.laplacian_solver.hodge_decomposition(filtered_frame)
            energies[index] = float(np.linalg.norm(coexact) ** 2)

        self.pot_estimator.fit_initial_window(energies)
        self.is_calibrated = True

    def process_photon_frame(
        self, raw_optical_frame: np.ndarray
    ) -> tuple[bool, np.ndarray, float]:
        """Filter one phase frame and classify its co-exact residual energy.

        Returns
        -------
        is_eavesdropping_attack : bool
            True when the incoming co-exact energy exceeds the current POT bound.
        reconstructed_phase_frame : numpy.ndarray
            Diffused, calibrated phase vector.
        coexact_energy : float
            Squared norm of the co-exact Hodge component.
        """
        if not self.is_calibrated:
            raise RuntimeError("Filter must be calibrated with ambient sky baseline.")

        whitened_frame = self.calibrator.process_frame(raw_optical_frame)
        reconstructed_phase = self.U_diffusion @ whitened_frame
        # Channel-wise baseline correction can shift a frame's spatial mean.
        # If that shift makes spatial variance larger, apply only the uniform
        # contraction needed to keep the filter from inventing frame energy.
        raw_variance = float(np.var(np.asarray(raw_optical_frame, dtype=float)))
        filtered_variance = float(np.var(reconstructed_phase))
        if filtered_variance > raw_variance and filtered_variance > 0.0:
            mean_phase = float(np.mean(reconstructed_phase))
            reconstructed_phase = mean_phase + (reconstructed_phase - mean_phase) * np.sqrt(
                raw_variance / filtered_variance
            )
        _, v_coexact, _ = self.laplacian_solver.hodge_decomposition(
            reconstructed_phase
        )
        energy = float(np.linalg.norm(v_coexact) ** 2)
        is_attack, _ = self.pot_estimator.update_dynamic_threshold(energy)
        return is_attack, reconstructed_phase, energy
