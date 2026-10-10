# Repository instructions

- Support Python 3.10 and newer; CI currently validates Python 3.10, 3.11, and 3.12.
- Use clear type annotations and NumPy-style docstrings for public classes and methods.
- Treat masks as `True` for observed values. Ignore masked entries and non-finite telemetry values in every score, fit, and detection path; never allow NaNs to become anomalies.
- Keep robust telemetry scoring based on median/MAD scale estimates, with documented fallback behavior for degenerate scales.
- Keep EVT thresholding based on a fitted Generalized Pareto tail and prevent detected anomalies from contaminating streaming calibration history.
- Preserve the homological chain constraint `d1 @ d2 ≈ 0` and use stable matrix-exponential diffusion.
- Add or update tests for behavioral changes, edge cases, and invalid inputs. Run `pytest --cov=src --cov-fail-under=90` before proposing changes; CI enforces at least 90% coverage.
- Avoid adding runtime dependencies unless they are necessary and documented in `pyproject.toml`.
