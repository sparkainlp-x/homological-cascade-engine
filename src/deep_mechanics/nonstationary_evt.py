"""Adaptive peaks-over-threshold estimation for drifting telemetry streams."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


class NonStationaryPOTEstimator:
    """Track a drifting baseline and fit a local Generalized Pareto tail.

    Clean observations update an exponentially weighted moving quantile and a
    bounded rolling calibration window. Detected extremes are deliberately
    excluded from that history. The fitted tail's shape and scale are smoothed
    exponentially between periodic rolling-window refits.

    Parameters
    ----------
    risk_level : float, default=1e-3
        Target probability of exceeding the returned one-step upper threshold.
    init_quantile : float, default=0.98
        Baseline quantile used as the peaks-over-threshold cutoff.
    alpha_decay : float, default=0.995
        Exponential decay factor for quantile weights and parameter smoothing.
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
        self.q = float(risk_level)
        self.init_quantile = float(init_quantile)
        self.alpha = float(alpha_decay)
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
        ages = np.arange(values.size - 1, -1, -1, dtype=float)
        weights = np.power(self.alpha, ages)
        order = np.argsort(values, kind="stable")
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, self.init_quantile * cumulative[-1]))
        return float(values[order[min(index, values.size - 1)]])

    def _fit_tail(self, smooth: bool) -> bool:
        """Fit a GPD to local positive excesses; return whether it succeeded."""
        if self.t0 is None or not self.window:
            return False
        values = np.asarray(self.window, dtype=float)
        excesses = values[values > self.t0] - self.t0
        if excesses.size < self.min_exceedances:
            return False
        try:
            shape, _, scale = genpareto.fit(excesses, floc=0)
            shape, scale = float(shape), float(scale)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                return False
        except (ValueError, RuntimeError, FloatingPointError):
            return False

        if smooth:
            self.gpd_gamma = self.alpha * self.gpd_gamma + (1.0 - self.alpha) * shape
            self.gpd_sigma = self.alpha * self.gpd_sigma + (1.0 - self.alpha) * scale
        else:
            self.gpd_gamma, self.gpd_sigma = shape, scale
        return True

    def fit_initial_window(self, baseline_data: np.ndarray) -> None:
        """Initialize the threshold and tail model from finite baseline values.

        Non-finite entries are ignored. At least one finite baseline value is
        required; when too few threshold exceedances are available, the shape
        default is retained and scale is initialized from the observed excesses
        or the baseline standard deviation.
        """
        data = np.asarray(baseline_data, dtype=float).reshape(-1)
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise ValueError("baseline_data must contain a finite value")
        if data.size > self.window_size:
            data = data[-self.window_size :]
        self.window = [float(value) for value in data]
        self.t0 = self._weighted_quantile(data)
        excesses = data[data > self.t0] - self.t0
        self.gpd_gamma = 0.1
        if excesses.size:
            self.gpd_sigma = max(
                float(np.median(excesses) / np.log(2.0)), np.finfo(float).eps
            )
        else:
            self.gpd_sigma = max(float(np.std(data)), np.finfo(float).eps)
        self._updates_since_refit = 0
        self._fit_tail(smooth=False)

    def _exceedance_rate(self) -> float:
        if self.t0 is None or not self.window:
            return 0.0
        values = np.asarray(self.window, dtype=float)
        return float(np.count_nonzero(values > self.t0) / values.size)

    def compute_extreme_value_bound(self, horizon: int = 1) -> float:
        """Return a GPD upper bound for a horizon-wide exceedance risk.

        A Bonferroni allocation uses ``risk_level / horizon`` per observation.
        If the empirical POT rate does not exceed that per-step risk, the
        threshold is the current baseline cutoff because tail extrapolation is
        not supported by the available calibration data.
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
        per_step_risk = self.q / int(horizon)
        if exceedance_rate <= per_step_risk:
            return float(self.t0)
        ratio = exceedance_rate / per_step_risk
        shape = float(self.gpd_gamma)
        scale = max(float(self.gpd_sigma), np.finfo(float).eps)
        if abs(shape) < 1e-6:
            excess_bound = scale * np.log(ratio)
        else:
            exponent = shape * np.log(ratio)
            if exponent > np.log(np.finfo(float).max):
                return float("inf")
            excess_bound = scale * np.expm1(exponent) / shape
        return float(self.t0 + max(0.0, float(excess_bound)))

    def update_dynamic_threshold(self, live_value: float) -> tuple[bool, float]:
        """Classify a sample, then update calibration only when it is clean.

        Non-finite live values are ignored and never reported as anomalies.
        The threshold returned is the value used to classify the current sample.
        """
        if self.t0 is None:
            raise RuntimeError("Estimator must be fitted with fit_initial_window().")
        try:
            value = float(live_value)
        except (TypeError, ValueError):
            raise ValueError("live_value must be a finite scalar") from None

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
            self._fit_tail(smooth=True)
            self._updates_since_refit = 0
        return False, threshold
