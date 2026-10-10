"""Streaming non-stationary Peak-Over-Threshold tail estimation."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import genpareto


class NonStationaryPOTEstimator:
    """Estimate an adaptive upper-tail threshold without learning from alarms.

    This is a numerical anomaly-scoring component, not a validated QKD security
    detector. Accepted observations are retained in a bounded rolling window;
    model parameters are updated smoothly at a fixed cadence. A 15% multiplicative
    margin on the fitted threshold cushions finite-window fit uncertainty.
    """

    _threshold_safety_factor = 1.15

    def __init__(
        self,
        risk_level: float = 1e-4,
        init_quantile: float = 0.95,
        min_exceedances: int = 5,
        history_size: int = 5000,
        update_interval: int = 20,
        adaptation_rate: float = 0.05,
    ) -> None:
        self.risk_level = float(risk_level)
        self.init_quantile = float(init_quantile)
        if not np.isfinite(self.init_quantile) or not 0 < self.init_quantile < 1:
            raise ValueError("init_quantile must be between 0 and 1")
        if not np.isfinite(self.risk_level) or not 0 < self.risk_level < 1 - self.init_quantile:
            raise ValueError("risk_level must be positive and smaller than the modeled tail fraction")
        if isinstance(min_exceedances, bool) or not isinstance(min_exceedances, (int, np.integer)) or min_exceedances < 3:
            raise ValueError("min_exceedances must be an integer of at least 3")
        if isinstance(history_size, bool) or not isinstance(history_size, (int, np.integer)) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        if isinstance(update_interval, bool) or not isinstance(update_interval, (int, np.integer)) or update_interval < 1:
            raise ValueError("update_interval must be a positive integer")
        self.adaptation_rate = float(adaptation_rate)
        if not np.isfinite(self.adaptation_rate) or not 0 < self.adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be in (0, 1]")

        self.min_exceedances = int(min_exceedances)
        self.history_size = int(history_size)
        self.update_interval = int(update_interval)
        self.threshold: float | None = None
        self.gamma: float | None = None
        self.sigma: float | None = None
        self.baseline: float | None = None
        self._tail_rate: float | None = None
        self._history = np.empty(0, dtype=float)
        self._accepted_since_fit = 0

    @staticmethod
    def _clean_sample(values: np.ndarray) -> np.ndarray:
        sample = np.asarray(values, dtype=float)
        if sample.ndim == 0:
            sample = sample.reshape(1)
        sample = sample.reshape(-1)
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            raise ValueError("at least one finite calibration energy is required")
        if np.any(sample < 0):
            raise ValueError("energy values must be non-negative")
        return sample

    def _estimate(self, sample: np.ndarray) -> tuple[float, float, float, float, float]:
        baseline = float(np.quantile(sample, self.init_quantile))
        excesses = sample[sample > baseline] - baseline
        if excesses.size < self.min_exceedances:
            raise ValueError(
                f"need at least {self.min_exceedances} exceedances; got {excesses.size}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gamma, _, sigma = genpareto.fit(excesses, floc=0.0)
        gamma, sigma = float(gamma), float(sigma)
        if not np.isfinite(gamma) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("GPD fit produced invalid parameters")
        tail_rate = excesses.size / sample.size
        conditional_risk = float(np.clip(self.risk_level / tail_rate, np.finfo(float).tiny, 1.0))
        threshold = self._threshold_safety_factor * (
            baseline + float(genpareto.isf(conditional_risk, gamma, loc=0.0, scale=sigma))
        )
        if not np.isfinite(threshold):
            raise ValueError("GPD fit produced a non-finite threshold")
        return baseline, gamma, sigma, tail_rate, threshold

    def fit_initial_window(self, energies: np.ndarray) -> "NonStationaryPOTEstimator":
        """Fit the initial GPD model from a finite calibration window."""
        sample = self._clean_sample(energies)
        baseline, gamma, sigma, tail_rate, threshold = self._estimate(sample)
        self.baseline = baseline
        self.gamma = gamma
        self.sigma = sigma
        self._tail_rate = tail_rate
        self.threshold = threshold
        self._history = sample[-self.history_size :].copy()
        self._accepted_since_fit = 0
        return self

    def _adapt(self) -> None:
        try:
            candidate = self._estimate(self._history)
        except ValueError:
            return
        rate = self.adaptation_rate
        baseline, gamma, sigma, tail_rate, _ = candidate
        self.baseline = (1 - rate) * float(self.baseline) + rate * baseline
        self.gamma = (1 - rate) * float(self.gamma) + rate * gamma
        self.sigma = max(np.finfo(float).eps, (1 - rate) * float(self.sigma) + rate * sigma)
        self._tail_rate = (1 - rate) * float(self._tail_rate) + rate * tail_rate
        conditional_risk = float(np.clip(self.risk_level / self._tail_rate, np.finfo(float).tiny, 1.0))
        excess_quantile = float(genpareto.isf(conditional_risk, self.gamma, loc=0.0, scale=self.sigma))
        updated_threshold = self._threshold_safety_factor * (self.baseline + excess_quantile)
        if np.isfinite(updated_threshold):
            self.threshold = float(updated_threshold)

    def update_dynamic_threshold(self, energy: float) -> tuple[bool, dict[str, float]]:
        """Score one energy, then adapt only from accepted, finite observations."""
        if self.threshold is None:
            raise RuntimeError("fit the estimator before streaming updates")
        value_array = np.asarray(energy, dtype=float)
        if value_array.ndim != 0:
            raise ValueError("energy must be a scalar")
        value = float(value_array)
        if not np.isfinite(value) or value < 0:
            raise ValueError("energy must be finite and non-negative")

        is_attack = value > self.threshold
        if not is_attack:
            self._history = np.append(self._history, value)
            if self._history.size > self.history_size:
                self._history = self._history[-self.history_size :]
            self._accepted_since_fit += 1
            if self._accepted_since_fit >= self.update_interval:
                self._adapt()
                self._accepted_since_fit = 0
        status = {
            "threshold": float(self.threshold),
            "gamma": float(self.gamma),
            "sigma": float(self.sigma),
        }
        return bool(is_attack), status
