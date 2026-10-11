# Homological Cascade Engine

[![CI Suite](https://github.com/sparkainlp-x/homological-cascade-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkainlp-x/homological-cascade-engine/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/sparkainlp-x/homological-cascade-engine/branch/main/graph/badge.svg)](https://codecov.io/gh/sparkainlp-x/homological-cascade-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### Project status

The Homological Cascade Engine is an experimental Python framework for exploring topological signal representations and robust anomaly scoring in multichannel telemetry. The repository contains prototype implementations for homological-cascade processing, MAD-based telemetry scoring, and Peaks-Over-Threshold (POT) thresholding, along with tests for selected software behaviors.

### Evidence and limitations

The current code and tests do not establish reliable false-alarm control under non-stationary drift, superiority to established baselines, or suitability for operational deployment. Synthetic tests demonstrate software behavior on the tested cases; they do not establish performance on real telemetry or real-world attacks.

Tokamak plasma triage, battery dendrite detection, and QKD attack detection are potential research directions—not validated capabilities of this repository. Claims about detection performance, false-alarm rates, or warning lead times in those domains require separate, domain-specific evaluation.

### Validation roadmap

The next step is a reproducible evaluation on public, labelled telemetry using a protocol fixed before testing. Report alarm burden, event-level recall, detection delay, calibration stability, comparisons with relevant baselines, and uncertainty. Keep training, calibration, and test periods separate; do not tune thresholds on the test period.

Hardware, security, and domain-specific claims require their own tests. Label results clearly as SYNTHETIC, REPORTED, TARGET, or UNRUN, as appropriate.

### Contribution invitation

Researchers and engineers are invited to inspect the implementation, reproduce the tests, and help evaluate the framework on appropriate datasets. Please treat results as exploratory until independently validated.

## Deep mechanics utilities

`src.deep_mechanics.HigherDimensionalLaplacian` accepts adjacent real boundary maps `d_k` and `d_(k+1)`, checks their chain condition, and constructs `L_k = d_k.T @ d_k + d_(k+1) @ d_(k+1).T`. It provides numerical Betti-number and spectral-gap estimates plus exact/co-exact/harmonic projections. Its tolerances are numerical cutoffs; they do not establish an exact topological model for measured data.

`src.deep_mechanics.NonStationaryPOTEstimator` tracks a recency-weighted baseline quantile and fits a Generalized Pareto tail on a bounded window. It excludes flagged observations from updates and computes horizon-adjusted, model-based upper bounds. These bounds depend on a locally stable tail model and are not calibrated false-positive guarantees. Validate the fit and thresholds against the intended data before operational use.

```python
import numpy as np

from src.deep_mechanics import HigherDimensionalLaplacian, NonStationaryPOTEstimator

# A small chain complex: d_1 @ d_2 = 0.
d1 = np.array([[1.0, -1.0, 0.0]])
d2 = np.array([[0.0], [0.0], [1.0]])
topology = HigherDimensionalLaplacian(d1, d2)
print(topology.compute_betti_number(), topology.compute_spectral_gap())

# Calibrate a streaming upper-tail model from baseline scores.
calibration = np.random.default_rng(10).standard_t(df=5, size=2_000)
tail = NonStationaryPOTEstimator(risk_level=1e-3)
tail.fit_initial_window(calibration)
is_anomaly, threshold_used = tail.update_dynamic_threshold(8.0)
```

## Install

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest --cov=src --cov-fail-under=90
```

## Quickstart

```python
import numpy as np

from src import EventCascadeOperator, HomologicalMembraneChain, RobustMADDetector

chain = HomologicalMembraneChain(n_0cells=32, n_1cells=64, n_2cells=32)
assert np.linalg.norm(chain.d1 @ chain.d2, ord=2) < 1e-12

initial = np.random.default_rng(7).normal(size=chain.n1)
state = EventCascadeOperator(chain).propagate_cascade(initial, steps=5)

# Telemetry has shape (observations, channels); True means observed.
telemetry = np.random.default_rng(8).standard_t(df=3, size=(2_000, 16))
observed = np.isfinite(telemetry)
detector = RobustMADDetector(threshold=4.0, top_k=2, block_size=8)
anomaly_mask = detector.detect(telemetry, mask=observed, axis=0)
```

Calibrate POT on non-negative anomaly magnitudes, then update it with each incoming batch:

```python
from src import POTThreshold

rng = np.random.default_rng(9)
calibration = rng.pareto(a=2.5, size=3_000)
pot = POTThreshold(tail_quantile=0.90, target_rate=0.01).fit(calibration)
new_batch = rng.pareto(a=2.5, size=64)
anomalies = pot.update(new_batch)
print(pot.threshold, anomalies)
```

A mask uses `True` for an observed value. Non-finite or masked samples are excluded from model fitting and never returned as anomalies. The first call to `POTThreshold.update` bootstraps from its input and returns an all-false mask; use `fit` explicitly when a separate calibration set is available.

## CI and license

GitHub Actions runs pytest with a 90% line-coverage gate on Python 3.10, 3.11, and 3.12. Codecov upload is configured; a repository `CODECOV_TOKEN` secret may be added to enable authenticated uploads. The source is distributed under the MIT License; see [LICENSE](LICENSE).
