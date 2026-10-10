"""Robust baseline and gain calibration for multichannel detector telemetry."""

from __future__ import annotations

import numpy as np


class AnalogSoundCalibrator:
    """Estimate per-channel detector offsets and relative noise gains.

    The calibration uses a median offset and a median-absolute-deviation noise
    estimate. Gains are normalized to the median channel noise scale (rather
    than to unit variance), so whitening balances channels without arbitrarily
    amplifying the whole sensor array. Extreme gain corrections are bounded.
    """

    def __init__(self, sample_rate_hz: float = 10_000.0, n_channels: int = 1) -> None:
        if isinstance(sample_rate_hz, (bool, np.bool_)):
            raise TypeError("sample_rate_hz must be a finite positive number")
        self.sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if isinstance(n_channels, (bool, np.bool_)) or not isinstance(
            n_channels, (int, np.integer)
        ):
            raise TypeError("n_channels must be a positive integer")
        if n_channels < 1:
            raise ValueError("n_channels must be positive")

        self.n_channels = int(n_channels)
        self.baseline_offset: np.ndarray | None = None
        self.noise_scale: np.ndarray | None = None
        self.gain: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether detector baseline parameters have been estimated."""
        return self.baseline_offset is not None

    def _coerce_block(self, telemetry: np.ndarray, *, allow_vector: bool) -> tuple[np.ndarray, bool]:
        try:
            block = np.asarray(telemetry, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("telemetry must be a numeric array") from None
        was_vector = block.ndim == 1
        if was_vector and allow_vector:
            block = block.reshape(1, -1)
        if block.ndim != 2 or block.shape[1] != self.n_channels:
            raise ValueError(
                f"telemetry must have shape (n_frames, {self.n_channels})"
                + (f" or ({self.n_channels},)" if allow_vector else "")
            )
        if block.shape[0] == 0:
            raise ValueError("telemetry must contain at least one frame")
        if not np.all(np.isfinite(block)):
            raise ValueError("telemetry must contain only finite values")
        return block, was_vector

    def calibrate_baseline(self, ambient_telemetry: np.ndarray) -> None:
        """Fit robust offsets and bounded, relative per-channel gains."""
        block, _ = self._coerce_block(ambient_telemetry, allow_vector=True)
        offset = np.median(block, axis=0)
        deviations = np.abs(block - offset)
        scale = 1.4826 * np.median(deviations, axis=0)
        positive_scales = scale[scale > np.finfo(float).eps]
        reference_scale = (
            float(np.median(positive_scales)) if positive_scales.size else 1.0
        )

        gain = np.ones(self.n_channels, dtype=float)
        usable = scale > np.finfo(float).eps
        gain[usable] = reference_scale / scale[usable]
        np.clip(gain, 0.25, 1.0, out=gain)

        self.baseline_offset = offset
        self.noise_scale = scale
        self.gain = gain

    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Remove the calibrated offset and balance detector channel gains."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be fitted with calibrate_baseline().")
        frame, was_vector = self._coerce_block(raw_frame, allow_vector=True)
        if not was_vector:
            raise ValueError("raw_frame must be a one-dimensional channel vector")
        assert self.baseline_offset is not None and self.gain is not None
        return (frame[0] - self.baseline_offset) * self.gain

    def process_block_whitened(self, telemetry: np.ndarray) -> np.ndarray:
        """Apply the calibrated offset and channel gains to a telemetry block."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be fitted with calibrate_baseline().")
        block, was_vector = self._coerce_block(telemetry, allow_vector=True)
        assert self.baseline_offset is not None and self.gain is not None
        whitened = (block - self.baseline_offset) * self.gain
        return whitened[0] if was_vector else whitened
