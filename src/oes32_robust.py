"""Robust MAD-scaled telemetry scoring with optional spatial Top-K filtering."""

from __future__ import annotations

import warnings

import numpy as np


class RobustMADDetector:
    """Score telemetry with median/MAD scaling and a Huber-loss transform.

    ``mask`` uses ``True`` for observed samples; false entries and non-finite
    values are ignored. ``axis`` is the observation axis. When ``top_k`` is
    provided, at most that many above-threshold features are retained in each
    contiguous flattened feature block of ``block_size``.

    Parameters
    ----------
    threshold : float
        Minimum robust Huber score considered anomalous.
    huber_delta : float
        Transition point between quadratic and linear Huber loss.
    top_k : int or None
        Maximum anomalies retained in each feature block, or no cap if None.
    block_size : int or None
        Number of flattened output features per block. Defaults to all features.
    """

    def __init__(
        self,
        threshold: float = 4.0,
        huber_delta: float = 1.345,
        top_k: int | None = None,
        block_size: int | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.huber_delta = float(huber_delta)
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if block_size is not None and (
            isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer or None")
        if block_size is not None and top_k is None:
            raise ValueError("block_size requires top_k")
        self.top_k = None if top_k is None else int(top_k)
        self.block_size = None if block_size is None else int(block_size)

    @staticmethod
    def _prepare(data: np.ndarray, mask: np.ndarray | None, axis: int) -> tuple[np.ndarray, np.ndarray, int]:
        values = np.asarray(data, dtype=float)
        if values.ndim == 0:
            raise ValueError("data must have at least one dimension")
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise TypeError("axis must be an integer")
        normalized_axis = int(axis)
        if normalized_axis < -values.ndim or normalized_axis >= values.ndim:
            raise ValueError("axis is out of bounds for data")
        normalized_axis %= values.ndim
        if values.size == 0 or values.shape[normalized_axis] == 0:
            raise ValueError("data must contain at least one observation")
        if mask is None:
            observed = np.ones(values.shape, dtype=bool)
        else:
            observed = np.array(mask, dtype=bool, copy=True)
            if observed.shape != values.shape:
                raise ValueError("mask must have the same shape as data")
        observed &= np.isfinite(values)
        return np.moveaxis(values, normalized_axis, 0), np.moveaxis(observed, normalized_axis, 0), normalized_axis

    def score(self, data: np.ndarray, mask: np.ndarray | None = None, axis: int = 0) -> np.ndarray:
        """Return sample-aligned robust Huber scores with the input shape."""
        values, observed, _ = self._prepare(data, mask, axis)
        working = np.where(observed, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(working, axis=0)
            mad = np.nanmedian(np.abs(working - center), axis=0)
            q25, q75 = np.nanpercentile(working, [25, 75], axis=0)
        scale = 1.4826 * mad
        fallback = (q75 - q25) / 1.349
        floor = np.finfo(float).eps * np.maximum(1.0, np.abs(np.nan_to_num(center)))
        scale = np.where(np.isfinite(scale) & (scale > floor), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
        center = np.nan_to_num(center, nan=0.0)
        z = np.divide(np.abs(values - center), scale, out=np.zeros_like(values), where=observed)
        delta = self.huber_delta
        loss = np.where(z <= delta, 0.5 * z**2, delta * (z - 0.5 * delta))
        scores = np.sqrt(2.0 * np.maximum(loss, 0.0))
        scores[~observed] = 0.0
        valid_features = np.any(observed, axis=0)
        scores = np.where(valid_features, scores, 0.0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
        return np.moveaxis(scores, 0, axis)

    def detect(self, data: np.ndarray, mask: np.ndarray | None = None, axis: int = 0) -> np.ndarray:
        """Return a sample-aligned boolean anomaly mask with the input shape."""
        _, observed, normalized_axis = self._prepare(data, mask, axis)
        scores = np.moveaxis(self.score(data, mask=mask, axis=axis), normalized_axis, 0)
        detections = (scores >= self.threshold) & observed
        if self.top_k is not None and detections.size:
            score_rows = scores.reshape(scores.shape[0], -1)
            detection_rows = detections.reshape(detections.shape[0], -1).copy()
            width = self.block_size or max(1, score_rows.shape[1])
            for row in range(score_rows.shape[0]):
                for start in range(0, score_rows.shape[1], width):
                    stop = min(start + width, score_rows.shape[1])
                    local = np.flatnonzero(detection_rows[row, start:stop])
                    if local.size > self.top_k:
                        order = np.lexsort((local, -score_rows[row, start:stop][local]))
                        keep = local[order[: self.top_k]]
                        detection_rows[row, start:stop] = False
                        detection_rows[row, start + keep] = True
            detections = detection_rows.reshape(detections.shape)
        return np.moveaxis(detections, 0, normalized_axis)
