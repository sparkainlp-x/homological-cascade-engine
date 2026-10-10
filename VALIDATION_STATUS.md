# Validation Status

This document separates what the repository implements from what remains a research hypothesis.

## Implemented (prototype code and software tests)
- Homological cascade / cell-complex topology code (`src/homological_cascade.py`)
- Robust telemetry scoring with MAD scaling and NaN-mask handling (`src/oes32_robust.py`)
- Peaks-over-threshold (POT/GPD) adaptive thresholding (`src/pot_threshold.py`)
- Experimental QKD/optics prototype (this branch)
- Unit tests; CI is configured with a 90% coverage gate. Verify the actual CI run and coverage on each commit before quoting a figure.

## Not implemented in this repository
- Tokamak plasma triage
- BESS battery dendrite detection
- Visual branding assets

## Not yet validated
- False-alarm control under non-stationary drift on real telemetry
- Comparison against strong baselines under a frozen protocol
- Any real-world attack detection: the synthetic QKD test checks software behaviour only
- Detection lead times or performance in tokamak, battery, or QKD settings

## Evidence labels
SYNTHETIC = simulated data only. REPORTED = taken from a cited source. TARGET = design goal, not a result. UNRUN = not yet executed.

## Next validation step
Evaluate on public labelled telemetry with a frozen protocol. Report alarm burden, event recall, detection delay, calibration stability, and uncertainty. Domain claims need separate domain-specific tests.
