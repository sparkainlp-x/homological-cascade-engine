"""Homological chain construction and diffusion over a finite 2-complex."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class HomologicalMembraneChain:
    """Construct a real-valued 2-complex chain with ``d1 @ d2 == 0``.

    The first boundary map is a seeded random matrix. The columns of the second
    map are selected from the right null space of the first, so the chain
    condition holds to floating-point precision. Coefficients are real rather
    than integral; this is a numerical model, not a combinatorial cellulation.

    Parameters
    ----------
    n_0cells, n_1cells, n_2cells : int
        Number of 0-, 1-, and 2-cells.
    seed : int
        Seed for reproducible construction of the first boundary map.
    """

    def __init__(
        self, n_0cells: int = 32, n_1cells: int = 64, n_2cells: int = 32,
        seed: int = 42,
    ) -> None:
        for name, value in (("n_0cells", n_0cells), ("n_1cells", n_1cells), ("n_2cells", n_2cells)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "n_2cells" else 1):
                raise ValueError(f"{name} is outside its allowed range")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")

        self.n0, self.n1, self.n2 = int(n_0cells), int(n_1cells), int(n_2cells)
        self.d1 = self._build_random_boundary(self.n0, self.n1, int(seed))
        self.d2 = self._build_kernel_boundary(self.d1, self.n2)

    @staticmethod
    def _build_random_boundary(rows: int, cols: int, seed: int = 42) -> np.ndarray:
        """Build a reproducible sparse signed boundary matrix."""
        rng = np.random.default_rng(seed)
        matrix = rng.choice([0, 1, -1], size=(rows, cols), p=[0.7, 0.15, 0.15])
        return matrix.astype(float)

    @staticmethod
    def _build_kernel_boundary(d1: np.ndarray, n2: int) -> np.ndarray:
        """Build ``n2`` columns in the numerical kernel of ``d1``."""
        boundary = np.asarray(d1, dtype=float)
        if boundary.ndim != 2:
            raise ValueError("d1 must be a two-dimensional matrix")
        if isinstance(n2, bool) or not isinstance(n2, (int, np.integer)):
            raise TypeError("n2 must be an integer")
        if n2 < 0:
            raise ValueError("n2 must be non-negative")

        _, singular_values, vh = np.linalg.svd(boundary, full_matrices=True)
        if singular_values.size:
            tolerance = max(boundary.shape) * np.finfo(float).eps * singular_values[0]
            rank = int(np.count_nonzero(singular_values > tolerance))
        else:
            rank = 0
        null_space = vh[rank:].T
        if null_space.shape[1] < n2:
            padding = np.zeros((boundary.shape[1], n2 - null_space.shape[1]))
            null_space = np.hstack((null_space, padding))
        return null_space[:, :n2].copy()


class EventCascadeOperator:
    """Diffuse perturbations over the combinatorial 1-Laplacian."""

    def __init__(self, membrane: HomologicalMembraneChain, coupling_strength: float = 0.85) -> None:
        coupling = float(coupling_strength)
        if not np.isfinite(coupling) or coupling < 0:
            raise ValueError("coupling_strength must be finite and non-negative")
        self.membrane = membrane
        self.J = coupling
        d1, d2 = membrane.d1, membrane.d2
        self.L1 = (d1.T @ d1) + (d2 @ d2.T)

    def propagate_cascade(
        self, initial_perturbation: np.ndarray, steps: int = 5, dt: float = 0.05
    ) -> np.ndarray:
        """Apply exact matrix-exponential diffusion for ``steps * dt`` time.

        The returned state is ``exp(-J * L1 * steps * dt) @ initial``. This
        semigroup form avoids the instability of explicit Euler stepping.
        """
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep < 0:
            raise ValueError("dt must be finite and non-negative")
        state = np.asarray(initial_perturbation, dtype=float)
        if state.shape != (self.membrane.n1,):
            raise ValueError(f"initial_perturbation must have shape ({self.membrane.n1},)")
        if not np.all(np.isfinite(state)):
            raise ValueError("initial_perturbation must contain only finite values")
        diffusion = expm(-self.J * self.L1 * (int(steps) * timestep))
        return diffusion @ state


class ObserverManifestOperator:
    """Project a continuous cascade state into a quantile-thresholded mask."""

    def __init__(self, threshold_quantile: float = 0.95) -> None:
        quantile = float(threshold_quantile)
        if not np.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("threshold_quantile must be between 0 and 1")
        self.tau_q = quantile

    def collapse_manifestation(self, cascade_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the high-energy mask and values, excluding non-finite entries."""
        state = np.asarray(cascade_state, dtype=float)
        if state.size == 0:
            raise ValueError("cascade_state must not be empty")
        valid = np.isfinite(state)
        if not np.any(valid):
            raise ValueError("cascade_state must contain at least one finite value")
        energy_density = np.abs(state[valid]) ** 2
        manifest_threshold = np.quantile(energy_density, self.tau_q)
        mask = np.zeros(state.shape, dtype=int)
        mask[valid] = (energy_density >= manifest_threshold).astype(int)
        values = np.where(mask.astype(bool), state, 0.0)
        return mask, values
