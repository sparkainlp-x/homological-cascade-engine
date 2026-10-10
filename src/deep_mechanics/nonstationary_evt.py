"""Adaptive peaks-over-threshold estimation for drifting telemetry streams."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import genpareto


class NonStationaryPOTEstimator:
    """Track a drifting baseline and fit a local Generalized Pareto tail.

    Clean observations update a recency-weighted baseline quantile and a bounded
    rolling calibration window. Detected extremes are deliberately excluded
    from that history. Tail parameters are smoothed between periodic refits.

    ``risk_level`` is a model-based upper-tail target, not a calibrated
    false-positive guarantee. The extrapolation assumes a locally stable GPD
    tail above the current baseline and should be validated for the data source.

    Parameters
    ----------
    risk_level : float, default=1e-3
        Target probability of exceeding the returned one-step upper threshold.
    init_quantile : float, default=0.98
        Baseline quantile used as the peaks-over-threshold cutoff.
    alpha_decay : float, default=0.995
        Exponential decay factor for observation weights and parameter smoothing.
        Must be in ``[0, 1)``; larger values retain more history.
    window_size : int, default=1000
        Maximum number of clean observations retained for local calibration.
    refit_interval : int, default=10
        Number of clean updates between GPD parameter refits.
    min_exceedances : int, default=5
        Minimum number of local exceedances required for a GPD refit.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_quantile: float = 0.98,
        alpha_decay: float = 0.995,
        window_size: int = 1000,
        refit_interval: int = 10,
        min_exceedances: int = 5,
    ) -> None:
        try:
            self.q = float(risk_level)
            self.init_quantile = float(init_quantile)
            self.alpha = float(alpha_decay)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("risk_level, init_quantile, and alpha_decay must be finite numbers") from exc
        if not np.isfinite(self.q) or not 0 < self.q < 1:
            raise ValueError("risk_level must be finite and between 0 and 1")
        if not np.isfinite(self.init_quantile) or not 0 < self.init_quantile < 1:
            raise ValueError("init_quantile must be finite and between 0 and 1")
        if self.q > 1.0 - self.init_quantile:
            raise ValueError("risk_level must not exceed the POT exceedance rate")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha < 1:
            raise ValueError("alpha_decay must be finite and in [0, 1)")
        for name, value in (
            ("window_size", window_size),
            ("refit_interval", refit_interval),
            ("min_exceedances", min_exceedances),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if min_exceedances < 2:
            raise ValueError("min_exceedances must be at least 2")

        self.window_size = int(window_size)
        self.refit_interval = int(refit_interval)
        self.min_exceedances = int(min_exceedances)
        self.window: list[float] = []
        self.t0: float | None = None
        self.gpd_gamma = 0.1
        self.gpd_sigma = 1.0
        self._updates_since_refit = 0

    def _weighted_quantile(self, data: np.ndarray) -> float:
        """Calculate an inverse-CDF quantile with exponentially decayed weights."""
        values = np.asarray(data, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("weighted quantile requires at least one finite value")
        ages = np.arange(values.size - 1, -1, -1, dtype=float)
        weights = np.power(self.alpha, ages)
        order = np.argsort(values, kind="stable")
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, self.init_quantile * cumulative[-1]))
        return float(values[order[min(index, values.size - 1)]])

    @staticmethod
    def _finite_excesses(values: np.ndarray, threshold: float) -> np.ndarray:
        """Compute finite positive excesses without leaking overflow warnings."""
        with np.errstate(over="ignore", invalid="ignore"):
            excesses = values[values > threshold] - threshold
        return excesses[np.isfinite(excesses) & (excesses > 0)]

    def _fit_tail(self, smooth: bool) -> bool:
        """Fit a GPD to local positive excesses; return whether it succeeded."""
        if self.t0 is None or not self.window:
            return False
        values = np.asarray(self.window, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        excesses = self._finite_excesses(values, self.t0)
        if excesses.size < self.min_exceedances:
            return False
        try:
            shape, _, scale = genpareto.fit(excesses, floc=0)
            shape, scale = float(shape), float(scale)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            return False
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            return False

        if smooth:
            self.gpd_gamma = self.alpha * self.gpd_gamma + (1.0 - self.alpha) * shape
            self.gpd_sigma = self.alpha * self.gpd_sigma + (1.0 - self.alpha) * scale
        else:
            self.gpd_gamma, self.gpd_sigma = shape, scale
        return bool(np.isfinite(self.gpd_gamma) and np.isfinite(self.gpd_sigma) and self.gpd_sigma > 0)

    def fit_initial_window(self, baseline_data: np.ndarray) -> None:
        """Initialize the baseline and tail model from finite calibration values.

        Non-finite entries are ignored. At least one finite value is required.
        If too few threshold exceedances are available, the default shape is
        retained and scale is initialized from observed excesses or the data
        spread so that a bound can still be computed without an unstable fit.
        """
        try:
            raw = np.asarray(baseline_data)
            if np.iscomplexobj(raw):
                raise ValueError("baseline_data must contain real values")
            data = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("baseline_data"):
                raise
            raise ValueError("baseline_data must be a real-valued numeric array") from exc
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise ValueError("baseline_data must contain a finite value")
        if data.size > self.window_size:
            data = data[-self.window_size :]
        baseline = self._weighted_quantile(data)
        excesses = self._finite_excesses(data, baseline)
        self.window = [float(value) for value in data]
        self.t0 = baseline
        self.gpd_gamma = 0.1
        if excesses.size:
            scale = float(np.median(excesses) / math.log(2.0))
        else:
            # Center before computing spread to reduce overflow from large offsets.
            with np.errstate(over="ignore", invalid="ignore"):
                centered = data - baseline
                scale = float(np.std(centered))
        if not np.isfinite(scale) or scale <= 0:
            scale = max(float(np.finfo(float).eps * max(1.0, abs(baseline))), np.finfo(float).tiny)
        self.gpd_sigma = scale
        self._updates_since_refit = 0
        self._fit_tail(smooth=False)

    def _exceedance_rate(self) -> float:
        if self.t0 is None or not self.window:
            return 0.0
        values = np.asarray(self.window, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        return float(np.count_nonzero(values > self.t0) / values.size)

    def compute_extreme_value_bound(self, horizon: int = 1) -> float:
        """Return a GPD upper bound for a horizon-wide exceedance risk.

        A Bonferroni allocation uses ``risk_level / horizon`` per observation.
        The computation uses logarithms and ``expm1`` to avoid overflow and
        cancellation in the tail-ratio formula. If the empirical POT rate is no
        greater than the per-step risk, the baseline cutoff is returned because
        tail extrapolation is not supported by the calibration data.
        """
        if self.t0 is None:
            raise RuntimeError("Estimator must be fitted with fit_initial_window().")
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, (int, np.integer)
        ):
            raise TypeError("horizon must be an integer")
        if horizon < 1:
            raise ValueError("horizon must be positive")

        exceedance_rate = self._exceedance_rate()
        if exceedance_rate == 0:
            return float(self.t0)
        try:
            log_ratio = (
                math.log(exceedance_rate)
                + math.log(int(horizon))
                - math.log(self.q)
            )
        except (OverflowError, ValueError):
            return float("inf")
        if log_ratio <= 0:
            return float(self.t0)

        shape = float(self.gpd_gamma)
        scale = float(self.gpd_sigma)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("fitted GPD parameters are invalid")
        exponent = shape * log_ratio
        if shape == 0.0:
            excess_bound = scale * log_ratio
        else:
            if shape > 0 and exponent >= math.log(np.finfo(float).max):
                return float("inf")
            try:
                excess_bound = scale * (math.expm1(exponent) / shape)
            except OverflowError:
                return float("inf")
        if not np.isfinite(excess_bound):
            return float("inf") if excess_bound > 0 else float(self.t0)
        threshold = float(self.t0) + max(0.0, excess_bound)
        return threshold if np.isfinite(threshold) else float("inf")

    def update_dynamic_threshold(self, live_value: float) -> tuple[bool, float]:
        """Classify a sample, then update calibration only when it is clean.

        Non-finite live values are ignored and never reported as anomalies. The
        returned threshold is the one used to classify the current sample; any
        calibration update affects later samples only.
        """
        if self.t0 is None:
            raise RuntimeError("Estimator must be fitted with fit_initial_window().")
        try:
            raw = np.asarray(live_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("live_value must be a finite scalar") from exc
        if raw.ndim != 0 or np.iscomplexobj(raw):
            raise ValueError("live_value must be a finite scalar")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("live_value must be a finite scalar") from exc

        threshold = self.compute_extreme_value_bound()
        if not np.isfinite(value):
            return False, threshold
        is_anomaly = bool(value > threshold)
        if is_anomaly:
            return True, threshold

        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.t0 = self._weighted_quantile(np.asarray(self.window, dtype=float))
        self._updates_since_refit += 1
        if self._updates_since_refit >= self.refit_interval:
            if self._fit_tail(smooth=True):
                self._updates_since_refit = 0
            else:
                # Retry after the next accepted sample rather than waiting for
                # another full interval while the parameters remain stale.
                self._updates_since_refit = self.refit_interval
        return False, threshold
