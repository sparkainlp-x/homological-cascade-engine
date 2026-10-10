"""Streaming Peak-Over-Threshold thresholds for robust anomaly masking."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import genpareto


class POTThreshold:
    """Fit an EVT Generalized Pareto tail and update it with accepted data.

    The estimator models the upper tail of non-negative anomaly magnitudes.
    ``fit`` calibrates the model; ``detect`` masks values above its dynamic
    threshold; ``update`` scores a batch using the current model and refits
    using only valid, non-anomalous observations. On the first ``update`` the
    batch initializes the model and is treated as calibration data.

    Parameters
    ----------
    tail_quantile : float
        Empirical quantile used as the POT baseline ``u``.
    target_rate : float
        Desired unconditional upper-tail probability for the learned threshold.
    min_exceedances : int
        Minimum number of observations above ``u`` required to fit the GPD.
    history_size : int or None
        Maximum accepted observations retained for streaming refits.
    """

    def __init__(
        self,
        tail_quantile: float = 0.95,
        target_rate: float = 0.01,
        min_exceedances: int = 10,
        history_size: int | None = 10000,
    ) -> None:
        self.tail_quantile = float(tail_quantile)
        self.target_rate = float(target_rate)
        if not np.isfinite(self.tail_quantile) or not 0 < self.tail_quantile < 1:
            raise ValueError("tail_quantile must be between 0 and 1")
        if not np.isfinite(self.target_rate) or not 0 < self.target_rate < 1 - self.tail_quantile:
            raise ValueError("target_rate must be positive and smaller than the modeled tail fraction")
        if isinstance(min_exceedances, bool) or not isinstance(min_exceedances, (int, np.integer)) or min_exceedances < 3:
            raise ValueError("min_exceedances must be an integer of at least 3")
        if history_size is not None and (
            isinstance(history_size, bool) or not isinstance(history_size, (int, np.integer)) or history_size < 1
        ):
            raise ValueError("history_size must be a positive integer or None")
        self.min_exceedances = int(min_exceedances)
        self.history_size = None if history_size is None else int(history_size)
        self.threshold_: float | None = None
        self.baseline_: float | None = None
        self.shape_: float | None = None
        self.scale_: float | None = None
        self._history = np.empty(0, dtype=float)

    @staticmethod
    def _valid_values(values: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(values, dtype=float)
        valid = np.isfinite(data)
        if mask is not None:
            supplied = np.asarray(mask, dtype=bool)
            if supplied.shape != data.shape:
                raise ValueError("mask must have the same shape as values")
            valid &= supplied
        return data, valid

    def fit(self, values: np.ndarray, mask: np.ndarray | None = None) -> "POTThreshold":
        """Fit a GPD to positive excesses above the empirical baseline."""
        data, valid = self._valid_values(values, mask)
        sample = data[valid].reshape(-1)
        if sample.size == 0:
            raise ValueError("at least one finite, unmasked calibration value is required")
        if np.any(sample < 0):
            raise ValueError("POT anomaly magnitudes must be non-negative")
        baseline = float(np.quantile(sample, self.tail_quantile))
        excesses = sample[sample > baseline] - baseline
        if excesses.size < self.min_exceedances:
            raise ValueError(
                f"need at least {self.min_exceedances} exceedances; got {excesses.size}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            shape, _, scale = genpareto.fit(excesses, floc=0.0)
        scale = float(scale)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("GPD fit produced invalid parameters")
        observed_tail_rate = excesses.size / sample.size
        conditional_rate = self.target_rate / observed_tail_rate
        conditional_rate = float(np.clip(conditional_rate, np.finfo(float).tiny, 1.0))
        threshold = baseline + float(genpareto.isf(conditional_rate, shape, loc=0.0, scale=scale))
        if not np.isfinite(threshold):
            raise ValueError("GPD fit produced a non-finite threshold")
        self.baseline_ = baseline
        self.shape_ = float(shape)
        self.scale_ = scale
        self.threshold_ = threshold
        self._history = sample.copy()
        self._trim_history()
        return self

    def _trim_history(self) -> None:
        if self.history_size is not None and self._history.size > self.history_size:
            self._history = self._history[-self.history_size :]

    @property
    def threshold(self) -> float:
        """The fitted anomaly threshold; raises before calibration."""
        if self.threshold_ is None:
            raise RuntimeError("fit the POT estimator before reading its threshold")
        return self.threshold_

    def detect(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Mark valid values strictly above the fitted threshold."""
        if self.threshold_ is None:
            raise RuntimeError("fit the POT estimator before detecting anomalies")
        data, valid = self._valid_values(values, mask)
        return valid & (data > self.threshold)

    def update(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Detect with the current model, then refit on accepted observations."""
        data, valid = self._valid_values(values, mask)
        sample = data[valid].reshape(-1)
        if np.any(sample < 0):
            raise ValueError("POT anomaly magnitudes must be non-negative")
        if self.threshold_ is None:
            self.fit(data, mask=valid)
            return np.zeros(data.shape, dtype=bool)

        anomalies = self.detect(data, mask=valid)
        accepted = data[valid & ~anomalies].reshape(-1)
        if accepted.size:
            self._history = np.concatenate((self._history, accepted))
            self._trim_history()
            try:
                self.fit(self._history)
            except ValueError as error:
                # Retain the previous valid model when a small history cannot
                # supply enough tail observations for a fresh GPD estimate.
                if "need at least" not in str(error):
                    raise
        return anomalies
