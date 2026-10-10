"""Combinatorial 1-Laplacian and numerical Hodge decomposition utilities."""

from __future__ import annotations

import numpy as np


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a numerically rank-revealing orthonormal basis for a matrix range."""
    if matrix.shape[1] == 0 or not matrix.size:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


class HigherDimensionalLaplacian:
    """Build ``L1 = d1.T @ d1 + d2 @ d2.T`` and split 1-chains.

    Parameters
    ----------
    d1, d2 : array-like
        Boundary maps with shapes ``(n0, n1)`` and ``(n1, n2)``. Their product
        must satisfy the chain condition to numerical precision.
    """

    def __init__(self, d1: np.ndarray, d2: np.ndarray) -> None:
        self.d1 = np.asarray(d1, dtype=float)
        self.d2 = np.asarray(d2, dtype=float)
        if self.d1.ndim != 2 or self.d2.ndim != 2:
            raise ValueError("d1 and d2 must both be two-dimensional")
        if self.d1.shape[1] != self.d2.shape[0]:
            raise ValueError("d1 and d2 dimensions do not form a chain")
        if not np.all(np.isfinite(self.d1)) or not np.all(np.isfinite(self.d2)):
            raise ValueError("boundary maps must contain only finite values")
        chain_error = np.linalg.norm(self.d1 @ self.d2, ord=2)
        scale = max(1.0, np.linalg.norm(self.d1, ord=2) * np.linalg.norm(self.d2, ord=2))
        if chain_error > 1e-8 * scale:
            raise ValueError("boundary maps must satisfy d1 @ d2 = 0")

        self.n_edges = self.d1.shape[1]
        self.L_k = self.d1.T @ self.d1 + self.d2 @ self.d2.T
        # Remove only floating-point asymmetry, preserving the PSD operator.
        self.L_k = (self.L_k + self.L_k.T) * 0.5
        self.L1 = self.L_k
        self._exact_basis = _orthonormal_basis(self.d1.T)
        self._coexact_basis = _orthonormal_basis(self.d2)

    def hodge_decomposition(self, one_chain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact, coexact, and harmonic projections of a 1-chain.

        The components sum to the input up to floating-point precision. The
        tuple ordering is ``(exact, coexact, harmonic)``.
        """
        vector = np.asarray(one_chain, dtype=float)
        if vector.shape != (self.n_edges,):
            raise ValueError(f"one_chain must have shape ({self.n_edges},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("one_chain must contain only finite values")
        exact = self._exact_basis @ (self._exact_basis.T @ vector)
        coexact = self._coexact_basis @ (self._coexact_basis.T @ vector)
        harmonic = vector - exact - coexact
        return exact, coexact, harmonic
