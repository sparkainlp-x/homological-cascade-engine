"""Combinatorial Laplacians and Hodge decomposition for finite chain complexes."""

from __future__ import annotations

import numpy as np


def _real_matrix(name: str, value: np.ndarray) -> np.ndarray:
    """Convert a real-valued matrix input to an independent float snapshot."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real values")
        matrix = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(name):
            raise
        raise ValueError(f"{name} must be a real-valued numeric matrix") from exc
    return matrix


def _spectral_norm(matrix: np.ndarray) -> float:
    """Return a spectral norm, including for empty rectangular matrices."""
    if not matrix.size or min(matrix.shape) == 0:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2))


def _orthonormal_basis(
    matrix: np.ndarray, reference_scale: float | None = None
) -> np.ndarray:
    """Return a rank-revealing orthonormal basis for the matrix range."""
    if matrix.shape[1] == 0 or not matrix.size:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    scale = singular_values[0]
    if reference_scale is not None:
        scale = max(scale, float(reference_scale))
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


class HigherDimensionalLaplacian:
    """Construct a k-th combinatorial Laplacian from adjacent boundary maps.

    The boundary matrices follow the convention ``d_k: C_k -> C_(k-1)`` and
    ``d_(k+1): C_(k+1) -> C_k``. The resulting operator is
    ``L_k = d_k.T @ d_k + d_(k+1) @ d_(k+1).T``. Inputs are copied so later
    mutation of caller-owned arrays cannot invalidate the operator.

    Parameters
    ----------
    d_k : array-like
        Real boundary matrix with shape ``(dim(C_(k-1)), dim(C_k))``.
    d_kplus1 : array-like
        Real boundary matrix with shape ``(dim(C_k), dim(C_(k+1)))``.
    chain_tol : float, default=1e-10
        Relative tolerance for the chain condition. The residual is compared
        with ``chain_tol * max(1, ||d_k||_2 * ||d_(k+1)||_2)``.

    Raises
    ------
    ValueError
        If a matrix is malformed or non-finite, dimensions are incompatible,
        the chain condition is violated, or the resulting operator overflows.
    """

    def __init__(
        self, d_k: np.ndarray, d_kplus1: np.ndarray, chain_tol: float = 1e-10
    ) -> None:
        self.d_k = _real_matrix("d_k", d_k)
        self.d_kp1 = _real_matrix("d_kplus1", d_kplus1)
        if self.d_k.ndim != 2 or self.d_kp1.ndim != 2:
            raise ValueError("boundary operators must be two-dimensional matrices")
        if not np.all(np.isfinite(self.d_k)) or not np.all(np.isfinite(self.d_kp1)):
            raise ValueError("boundary operators must contain only finite values")
        if self.d_k.shape[1] != self.d_kp1.shape[0]:
            raise ValueError(
                "d_k columns must match d_kplus1 rows (the dimension of C_k)"
            )
        try:
            tolerance = float(chain_tol)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("chain_tol must be finite and non-negative") from exc
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("chain_tol must be finite and non-negative")

        with np.errstate(over="ignore", invalid="ignore"):
            composition = self.d_k @ self.d_kp1
            residual = float(np.linalg.norm(composition))
            scale = max(
                1.0,
                _spectral_norm(self.d_k) * _spectral_norm(self.d_kp1),
            )
            allowed_residual = tolerance * scale
        if not np.isfinite(residual) or residual > allowed_residual:
            raise ValueError(
                "Boundary chain property violated: "
                f"||d_k @ d_(k+1)|| = {residual:.2e}"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            laplacian = self.d_k.T @ self.d_k + self.d_kp1 @ self.d_kp1.T
            laplacian = (laplacian + laplacian.T) * 0.5
        if not np.all(np.isfinite(laplacian)):
            raise ValueError("boundary operators produce a non-finite Laplacian")

        self.chain_tol = tolerance
        self.L_k = laplacian
        # Keep the public structural arrays consistent with the cached operator.
        self.d_k.setflags(write=False)
        self.d_kp1.setflags(write=False)
        self.L_k.setflags(write=False)

    @property
    def dimension(self) -> int:
        """Return the dimension of the k-chain space."""
        return int(self.d_k.shape[1])

    @staticmethod
    def _validate_zero_tolerance(zero_tol: float) -> float:
        try:
            tolerance = float(zero_tol)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("zero_tol must be finite and non-negative") from exc
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

        Returns ``(v_exact, v_coexact, h)`` for the images of ``d_(k+1)`` and
        ``d_k.T`` and the numerically harmonic eigenspace of ``L_k``. Modes
        classified as harmonic by ``zero_tol`` are removed from the boundary
        ranges before projecting, preventing near-zero modes from being counted
        twice. For a valid chain complex the components are mutually orthogonal
        and sum to the input up to floating-point precision.
        """
        tolerance = self._validate_zero_tolerance(zero_tol)
        vector = _real_matrix("chain_vector", chain_vector)
        if vector.shape != (self.dimension,):
            raise ValueError(f"chain_vector must have shape ({self.dimension},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("chain_vector must contain only finite values")

        eigenvalues, eigenvectors = np.linalg.eigh(self.L_k)
        harmonic_basis = eigenvectors[:, np.abs(eigenvalues) <= tolerance]
        harmonic = (
            harmonic_basis @ (harmonic_basis.T @ vector)
            if harmonic_basis.shape[1]
            else np.zeros_like(vector)
        )

        # Restrict boundary ranges to the orthogonal complement of the chosen
        # numerical kernel; otherwise a user-supplied zero_tol can double-count
        # small positive eigenmodes as both boundary and harmonic components.
        boundary_scale = max(
            _spectral_norm(self.d_k), _spectral_norm(self.d_kp1)
        )
        exact_range = self.d_kp1
        coexact_range = self.d_k.T
        if harmonic_basis.shape[1]:
            exact_range = exact_range - harmonic_basis @ (harmonic_basis.T @ exact_range)
            coexact_range = coexact_range - harmonic_basis @ (
                harmonic_basis.T @ coexact_range
            )
        exact_basis = _orthonormal_basis(
            exact_range, reference_scale=boundary_scale
        )
        coexact_basis = _orthonormal_basis(
            coexact_range, reference_scale=boundary_scale
        )
        exact = exact_basis @ (exact_basis.T @ vector)
        coexact = coexact_basis @ (coexact_basis.T @ vector)
        return exact, coexact, harmonic

    def verify_boundary_orthogonality(
        self, chain_vector: np.ndarray, zero_tol: float = 1e-8, atol: float = 1e-8
    ) -> bool:
        """Check pairwise orthogonality of the three Hodge components.

        Dot products are compared against ``atol * max(1, ||a|| ||b||)`` to
        make the check meaningful for components at different scales.
        """
        try:
            tolerance = float(atol)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("atol must be finite and non-negative") from exc
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
