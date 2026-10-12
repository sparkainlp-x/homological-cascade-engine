"""Per-channel robust baseline calibration and whitening for frame streams.

This module is plain classical signal conditioning. It estimates a per-channel
centre (median) and a robust scale (MAD) from a calibration block and then
rescales later frames with those fixed statistics. It does not model any
physical sensor, analog front end, or acoustic hardware; the class name is
kept for API compatibility with the project specification.
"""

from __future__ import annotations

import numpy as np

_MAD_TO_SIGMA = 1.4826


class AnalogSoundCalibrator:
    """Robust per-channel centring and scaling of multichannel frames.

    Parameters
    ----------
    sample_rate_hz : float
        Nominal frame rate of the stream. It is validated and stored as
        metadata only; no time-domain filtering is performed.
    n_channels : int
        Number of channels per frame.

    Notes
    -----
    ``calibrate_baseline`` ignores non-finite calibration values. If a
    channel's MAD is zero (or the channel has no finite values), the scale falls
    back to the channel's standard deviation and then to ``1.0`` so that the
    whitening step never divides by zero.
    """

    def __init__(self, sample_rate_hz: float = 10_000.0, n_channels: int = 256) -> None:
        try:
            rate = float(sample_rate_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sample_rate_hz must be a finite positive number") from exc
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be a finite positive number")
        if isinstance(n_channels, (bool, np.bool_)) or not isinstance(
            n_channels, (int, np.integer)
        ):
            raise TypeError("n_channels must be an integer")
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.sample_rate_hz = rate
        self.n_channels = int(n_channels)
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        """Return whether baseline statistics are available."""
        return self.center is not None and self.scale is not None

    def _as_block(self, block: np.ndarray, name: str) -> np.ndarray:
        data = np.asarray(block)
        if np.iscomplexobj(data):
            raise ValueError(f"{name} must contain real values")
        try:
            data = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if data.ndim != 2 or data.shape[1] != self.n_channels:
            raise ValueError(f"{name} must have shape (n_frames, {self.n_channels})")
        return data

    def calibrate_baseline(self, block: np.ndarray) -> None:
        """Estimate per-channel median and MAD-based scale from ``block``.

        Parameters
        ----------
        block : array-like, shape (n_frames, n_channels)
            Calibration frames. Non-finite values are ignored.
        """
        data = self._as_block(block, "block")
        if data.shape[0] < 2:
            raise ValueError("block must contain at least two frames")
        finite = np.isfinite(data)
        if not np.any(finite):
            raise ValueError("block must contain finite values")
        masked = np.where(finite, data, np.nan)
        center = np.zeros(self.n_channels)
        scale = np.ones(self.n_channels)
        for channel in range(self.n_channels):
            values = masked[:, channel]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * _MAD_TO_SIGMA
            if not np.isfinite(mad) or mad <= 0:
                mad = float(np.std(values))
            if not np.isfinite(mad) or mad <= 0:
                mad = 1.0
            center[channel] = median
            scale[channel] = mad
        self.center = center
        self.scale = scale

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return ``(frame - center) / scale`` for one finite frame."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be fitted with calibrate_baseline().")
        values = np.asarray(frame)
        if np.iscomplexobj(values):
            raise ValueError("frame must contain real values")
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_channels,):
            raise ValueError(f"frame must have shape ({self.n_channels},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("frame must contain only finite values")
        return (values - self.center) / self.scale

    def process_block_whitened(self, block: np.ndarray) -> np.ndarray:
        """Apply :meth:`process_frame` to every row of a finite block."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be fitted with calibrate_baseline().")
        data = self._as_block(block, "block")
        if not np.all(np.isfinite(data)):
            raise ValueError("block must contain only finite values")
        return (data - self.center) / self.scale
