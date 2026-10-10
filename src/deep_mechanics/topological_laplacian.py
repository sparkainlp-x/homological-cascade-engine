"""Combinatorial Laplacians and Hodge decomposition for finite chain complexes."""

from __future__ import annotations

import numpy as np


class HigherDimensionalLaplacian:
    """Construct a k-th combinatorial Laplacian from adjacent boundary maps.

    The boundary matrices follow the convention ``d_k: C_k -> C_(k-1)`` and
    ``d_(k+1): C_(k+1) -> C_k``. The resulting Laplacian is
    ``d_k.T @ d_k + d_(k+1) @ d_(k+1).T``.

    Parameters
    ----------
    d_k : numpy.ndarray
        Boundary matrix with shape ``(dim(C_(k-1)), dim(C_k))``.
    d_kplus1 : numpy.ndarray
        Boundary matrix with shape ``(dim(C_k), dim(C_(k+1)))``.
    chain_tol : float, default=1e-10
        Absolute tolerance for the chain condition ``d_k @ d_(k+1) == 0``.

    Raises
    ------
    ValueError
        If either matrix has an invalid shape, contains non-finite values, or
        violates the chain condition.
    """

    def __init__(
        self, d_k: np.ndarray, d_kplus1: np.ndarray, chain_tol: float = 1e-10
    ) -> None:
        self.d_k = np.asarray(d_k, dtype=float)
        self.d_kp1 = np.asarray(d_kplus1, dtype=float)
        if self.d_k.ndim != 2 or self.d_kp1.ndim != 2:
            raise ValueError("boundary operators must be two-dimensional matrices")
        if not np.all(np.isfinite(self.d_k)) or not np.all(np.isfinite(self.d_kp1)):
            raise ValueError("boundary operators must contain only finite values")
        if self.d_k.shape[1] != self.d_kp1.shape[0]:
            raise ValueError(
                "d_k columns must match d_kplus1 rows (the dimension of C_k)"
            )
        tolerance = float(chain_tol)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("chain_tol must be finite and non-negative")
        boundary_residual = np.linalg.norm(self.d_k @ self.d_kp1)
        if boundary_residual > tolerance:
            raise ValueError(
                "Boundary chain property violated: "
                f"||d_k @ d_(k+1)|| = {boundary_residual:.2e}"
            )

        self.chain_tol = tolerance
        self.L_k = self.d_k.T @ self.d_k + self.d_kp1 @ self.d_kp1.T

    @property
    def dimension(self) -> int:
        """Return the dimension of the k-chain space."""
        return int(self.d_k.shape[1])

    @staticmethod
    def _validate_zero_tolerance(zero_tol: float) -> float:
        tolerance = float(zero_tol)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("zero_tol must be finite and non-negative")
        return tolerance

    def compute_betti_number(self, zero_tol: float = 1e-8) -> int:
        """Return ``dim(ker(L_k))`` using an absolute eigenvalue tolerance."""
        tolerance = self._validate_zero_tolerance(zero_tol)
        eigenvalues = np.linalg.eigvalsh(self.L_k)
        return int(np.count_nonzero(np.abs(eigenvalues) <= tolerance))

    def compute_spectral_gap(self, zero_tol: float = 1e-8) -> float:
        """Return the smallest positive Laplacian eigenvalue, or ``0.0``.

        Eigenvalues no larger than ``zero_tol`` are treated as numerical zeros.
        A zero value is returned when the entire k-chain space is harmonic.
        """
        tolerance = self._validate_zero_tolerance(zero_tol)
        eigenvalues = np.linalg.eigvalsh(self.L_k)
        positive = eigenvalues[eigenvalues > tolerance]
        return float(positive[0]) if positive.size else 0.0

    def hodge_decomposition(
        self, chain_vector: np.ndarray, zero_tol: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose a k-chain into exact, co-exact, and harmonic components.

        Returns components ``(v_exact, v_coexact, h)`` satisfying
        ``v = v_exact + v_coexact + h`` up to numerical precision, with
        ``v_exact`` in ``im(d_(k+1))``, ``v_coexact`` in ``im(d_k.T)``, and
        ``h`` in ``ker(L_k)``.
        """
        tolerance = self._validate_zero_tolerance(zero_tol)
        vector = np.asarray(chain_vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"chain_vector must have shape ({self.dimension},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("chain_vector must contain only finite values")

        eigenvalues, eigenvectors = np.linalg.eigh(self.L_k)
        harmonic_basis = eigenvectors[:, np.abs(eigenvalues) <= tolerance]
        if harmonic_basis.shape[1]:
            harmonic = harmonic_basis @ (harmonic_basis.T @ vector)
        else:
            harmonic = np.zeros_like(vector)

        exact_basis = self.d_kp1
        coexact_basis = self.d_k.T
        exact = exact_basis @ (np.linalg.pinv(exact_basis) @ vector)
        coexact = coexact_basis @ (np.linalg.pinv(coexact_basis) @ vector)
        return exact, coexact, harmonic

    def verify_boundary_orthogonality(
        self, chain_vector: np.ndarray, zero_tol: float = 1e-8, atol: float = 1e-8
    ) -> bool:
        """Check pairwise orthogonality of the three Hodge components.

        Dot products are compared against ``atol * max(1, ||a|| ||b||)`` to
        make the check meaningful for components at different scales.
        """
        tolerance = float(atol)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("atol must be finite and non-negative")
        exact, coexact, harmonic = self.hodge_decomposition(chain_vector, zero_tol)
        components = (exact, coexact, harmonic)
        for index, left in enumerate(components):
            for right in components[index + 1 :]:
                scale = max(1.0, float(np.linalg.norm(left) * np.linalg.norm(right)))
                if abs(float(np.dot(left, right))) > tolerance * scale:
                    return False
        return True
