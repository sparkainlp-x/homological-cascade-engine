"""Robust per-channel detector baseline and scale calibration."""

from __future__ import annotations

import numpy as np


class AnalogSoundCalibrator:
    """Estimate robust offsets and gains for multichannel sensor blocks.

    The historical class name is retained for compatibility with existing
    experimental code. Calibration uses the median and MAD, falling back to an
    interquartile scale and finally unit scale for degenerate channels.
    """

    def __init__(self, sample_rate_hz: float, n_channels: int) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)) or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        self.n_channels = int(n_channels)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether per-channel center and scale estimates are available."""
        return self.center_ is not None and self.scale_ is not None

    def calibrate_baseline(self, telemetry: np.ndarray) -> None:
        """Fit robust per-channel center and scale estimates from a data block."""
        values = np.asarray(telemetry, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(f"telemetry must have shape (n_samples, {self.n_channels})")
        if values.shape[0] < 2:
            raise ValueError("at least two calibration samples are required")
        valid_counts = np.sum(np.isfinite(values), axis=0)
        if np.any(valid_counts < 2):
            raise ValueError("each channel needs at least two finite calibration samples")
        with np.errstate(all="ignore"):
            center = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            mad = np.nanmedian(np.abs(values - center), axis=0)
            q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        scale = 1.4826 * mad
        floor = np.finfo(float).eps * np.maximum(1.0, np.abs(np.nan_to_num(center)))
        fallback = (q75 - q25) / 1.349
        scale = np.where(np.isfinite(scale) & (scale > floor), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > floor), scale, 1.0)
        self.center_ = np.asarray(center, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def _require_calibrated(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_calibrated:
            raise RuntimeError("calibrate_baseline must be called first")
        return self.center_, self.scale_

    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Whiten a frame and remove its robust common-mode offset.

        Missing entries map to zero before the median common-mode estimate, so
        non-finite sensor readings cannot create large spatial anomalies.
        """
        center, scale = self._require_calibrated()
        frame = np.asarray(raw_frame, dtype=float)
        if frame.shape != (self.n_channels,):
            raise ValueError(f"raw_frame must have shape ({self.n_channels},)")
        cleaned = np.where(np.isfinite(frame), frame, center)
        whitened = (cleaned - center) / scale
        whitened -= np.median(whitened)
        return np.nan_to_num(whitened, nan=0.0, posinf=0.0, neginf=0.0)

    def process_block_whitened(self, telemetry: np.ndarray) -> np.ndarray:
        """Whiten a block with the fitted center and channel scales."""
        self._require_calibrated()
        values = np.asarray(telemetry, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(f"telemetry must have shape (n_samples, {self.n_channels})")
        return np.vstack([self.process_frame(row) for row in values])

    def restore_scale(self, centered_frame: np.ndarray) -> np.ndarray:
        """Return a centered residual to sensor units without restoring offset."""
        _, scale = self._require_calibrated()
        values = np.asarray(centered_frame, dtype=float)
        if values.shape != (self.n_channels,):
            raise ValueError(f"centered_frame must have shape ({self.n_channels},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("centered_frame must contain only finite values")
        return values * scale
