"""Hodge-projected residual-energy scoring of synthetic phase-vector frames.

Scope (read this first)
-----------------------
This is a **classical, synthetic** signal-processing module. It operates on
real-valued vectors ("phase frames") supplied by the caller, typically
generated in tests. It is **not** a quantum key distribution (QKD) system, it
is **not** adaptive-optics hardware or a wavefront controller, it does not
process photons or quantum states, it gives **no quantum-security guarantee**,
and it does **not** detect real eavesdroppers. No latency has been measured.

Background (motivation only, not a claim)
-----------------------------------------
Free-space QKD links suffer from atmospheric turbulence, and mechanical
adaptive optics typically corrects at roughly kHz bandwidths while QKD clock
rates are far higher. That gap motivated asking whether a software-only
residual score can separate slow, smooth disturbances from sharp localized
ones in a synthetic setting. This module is a toy for exploring that question;
it does not model turbulence physics, Lindblad dynamics, or any attack.

What the code does
------------------
1. Robustly centre and scale each channel with statistics from a calibration
   block (:class:`src.analog_calibration.AnalogSoundCalibrator`).
2. Optionally remove the frame mean ("piston"). A disturbance that is common
   to every channel is not orthogonal to the co-exact subspace of the random
   chain complex used here, so without this step slow common-mode drift leaks
   into the score and raises false alarms.
3. Apply the linear operator ``U = expm(-J * L1 * dt)``, where ``L1`` is the
   1-Laplacian of :class:`src.homological_cascade.HomologicalMembraneChain`.
   ``L1`` is symmetric positive semi-definite, so ``U`` is symmetric with
   eigenvalues in ``(0, 1]``: a contraction that damps the exact and co-exact
   components and leaves the harmonic component unchanged.
4. Score each frame by the squared norm of its co-exact Hodge component.
5. Flag frames whose score exceeds a streaming Generalized Pareto
   peaks-over-threshold bound
   (:class:`src.deep_mechanics.NonStationaryPOTEstimator`).

A flag means only "this frame's co-exact residual energy is unusually large
relative to the calibration data". Any sufficiently large perturbation with a
co-exact component is flagged, whatever caused it; the flag is not specific to
tampering. The POT ``risk_level`` is a model-based target, not a calibrated
false-alarm guarantee.

The class and method names (``TopologicalAdaptiveOpticsFilter``,
``calibrate_ambient_sky``, ``process_photon_frame``) are kept for API
compatibility with the project specification and are descriptive labels only.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..analog_calibration import AnalogSoundCalibrator
from ..deep_mechanics.nonstationary_evt import NonStationaryPOTEstimator
from ..deep_mechanics.topological_laplacian import HigherDimensionalLaplacian
from ..homological_cascade import HomologicalMembraneChain


class TopologicalAdaptiveOpticsFilter:
    """Calibrated co-exact residual-energy scorer for synthetic frames.

    Parameters
    ----------
    n_subapertures : int, default=256
        Frame length; used as the number of 1-cells of the chain complex
        (32 0-cells and 128 2-cells, as in the specification).
    sample_rate_hz : float, default=10000.0
        Nominal frame rate, stored as metadata by the calibrator only.
    coupling_j : float, default=1.2
        Non-negative diffusion coupling ``J``.
    dt : float, default=0.02
        Non-negative diffusion time step.
    remove_piston : bool, default=True
        Subtract each conditioned frame's mean before diffusion.
    risk_level, init_quantile : float
        Passed to :class:`NonStationaryPOTEstimator`.
    seed : int, default=42
        Seed of the random boundary map of the chain complex.
    """

    def __init__(
        self,
        n_subapertures: int = 256,
        sample_rate_hz: float = 10_000.0,
        coupling_j: float = 1.2,
        *,
        dt: float = 0.02,
        remove_piston: bool = True,
        risk_level: float = 1e-4,
        init_quantile: float = 0.95,
        seed: int = 42,
    ) -> None:
        if isinstance(n_subapertures, (bool, np.bool_)) or not isinstance(
            n_subapertures, (int, np.integer)
        ):
            raise TypeError("n_subapertures must be an integer")
        if n_subapertures < 2:
            raise ValueError("n_subapertures must be at least 2")
        try:
            coupling = float(coupling_j)
            step = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coupling_j and dt must be finite numbers") from exc
        if not np.isfinite(coupling) or coupling < 0:
            raise ValueError("coupling_j must be finite and non-negative")
        if not np.isfinite(step) or step < 0:
            raise ValueError("dt must be finite and non-negative")

        self.n_channels = int(n_subapertures)
        self.J = coupling
        self.dt = step
        self.remove_piston = bool(remove_piston)
        self.calibrator = AnalogSoundCalibrator(sample_rate_hz, self.n_channels)
        self.membrane = HomologicalMembraneChain(
            n_0cells=32, n_1cells=self.n_channels, n_2cells=128, seed=seed
        )
        self.laplacian_solver = HigherDimensionalLaplacian(self.membrane.d1, self.membrane.d2)
        diffusion = expm(-self.J * np.asarray(self.laplacian_solver.L_k) * self.dt)
        self.U_diffusion = (diffusion + diffusion.T) * 0.5
        self.U_diffusion.setflags(write=False)
        self.pot_estimator = NonStationaryPOTEstimator(
            risk_level=risk_level, init_quantile=init_quantile
        )
        self.is_calibrated = False

    def _remove_piston(self, values: np.ndarray) -> np.ndarray:
        if not self.remove_piston:
            return values
        return values - values.mean(axis=-1, keepdims=True)

    def _coexact_energy(self, reconstructed: np.ndarray) -> float:
        _, coexact, _ = self.laplacian_solver.hodge_decomposition(reconstructed)
        return float(np.dot(coexact, coexact))

    def condition_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Return the calibrated (and optionally piston-removed) frame.

        This is the exact vector to which ``U_diffusion`` is applied.
        """
        if not self.calibrator.is_calibrated:
            raise RuntimeError("Filter must be calibrated with calibrate_ambient_sky().")
        return self._remove_piston(self.calibrator.process_frame(raw_frame))

    def calibrate_ambient_sky(self, ambient_optical_telemetry: np.ndarray) -> None:
        """Fit channel statistics and the score's tail model on a baseline block.

        Parameters
        ----------
        ambient_optical_telemetry : array-like, shape (n_frames, n_subapertures)
            Finite baseline frames assumed free of the disturbances to be
            flagged. At least two frames are required.
        """
        self.calibrator.calibrate_baseline(ambient_optical_telemetry)
        conditioned = self._remove_piston(
            self.calibrator.process_block_whitened(ambient_optical_telemetry)
        )
        energies = [
            self._coexact_energy(self.U_diffusion @ frame) for frame in conditioned
        ]
        self.pot_estimator.fit_initial_window(np.asarray(energies))
        self.is_calibrated = True

    def process_photon_frame(
        self, raw_optical_frame: np.ndarray
    ) -> tuple[bool, np.ndarray, float]:
        """Score one synthetic frame.

        Returns
        -------
        flagged : bool
            Whether the co-exact energy exceeded the current POT bound. Named
            ``is_eavesdropping_attack`` in the original specification; it is
            only a statistical exceedance flag.
        reconstructed : ndarray, shape (n_subapertures,)
            ``U_diffusion @ condition_frame(raw_optical_frame)``, in
            calibrated (dimensionless) units.
        coexact_energy : float
            Squared norm of the co-exact Hodge component of ``reconstructed``.

        Raises
        ------
        RuntimeError
            If the filter has not been calibrated.
        ValueError
            If the frame has the wrong shape or contains non-finite values.
        """
        if not self.is_calibrated:
            raise RuntimeError("Filter must be calibrated with calibrate_ambient_sky().")
        reconstructed = self.U_diffusion @ self.condition_frame(raw_optical_frame)
        energy = self._coexact_energy(reconstructed)
        flagged, _ = self.pot_estimator.update_dynamic_threshold(energy)
        return bool(flagged), reconstructed, energy
