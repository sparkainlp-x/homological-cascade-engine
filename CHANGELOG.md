# Changelog

## v0.6.0 (2026-10-11)

### Added
- `src.quantum.topological_ao.TopologicalAdaptiveOpticsFilter`: a **classical, synthetic** co-exact residual-energy scorer. It calibrates per-channel statistics, removes the frame mean (piston), applies `U = expm(-J * L1 * dt)`, scores the squared norm of the co-exact Hodge component, and flags exceedances of a streaming POT bound. Class and method names follow the project specification; they are labels only. This is not a QKD system, not adaptive-optics hardware, gives no quantum-security guarantee, does not detect real eavesdroppers, and has no measured latency.
- `src.analog_calibration.AnalogSoundCalibrator`: per-channel median/MAD centring and scaling with documented fallbacks. It models no physical sensor or front end.
- `tests/test_topological_ao.py`: 21 tests (the specification scenario, uncalibrated and shape/finiteness errors, determinism, the symmetric-contraction property of `U`, a bounded false-alarm count on pure noise, piston removal, and the fact that the flag responds to energy, not tampering specifically).

### Changed
- `HigherDimensionalLaplacian.hodge_decomposition` caches its projection bases per `zero_tol` (results are unchanged; repeated calls no longer redo the eigen- and singular-value decompositions).
- Citation metadata now names the author with ORCID; added `.zenodo.json`.

### Notes on the specification
- The specification asserted `var(reconstructed) <= var(raw_frame)`. That is false: the calibrator divides by a robust scale (about 0.67 in the test), so calibrated frames have larger variance than raw frames (it failed on 200 of 200 frames). The test now checks `var(reconstructed) <= var(conditioned_frame)`, which holds because `U` is a symmetric contraction applied to a zero-mean vector.
- Without piston removal the slow common-mode drift in the test leaks into the co-exact subspace (the random boundary map does not annihilate the all-ones vector) and caused 4 false alarms on frames 200–399. With piston removal there were 0, and the synthetic localized offset (stand-in for a tamper scenario, +18 on channels 50–99) scored 158.5 against a bound of about 2.7. `remove_piston=False` restores the unadjusted behaviour.
- All of the above is SYNTHETIC.
