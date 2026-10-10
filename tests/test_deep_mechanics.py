"""Verification for higher-dimensional topology and non-stationary EVT."""

import numpy as np
import pytest

from src.deep_mechanics import HigherDimensionalLaplacian, NonStationaryPOTEstimator
from src.homological_cascade import HomologicalMembraneChain


def test_hodge_decomposition_orthogonality():
    membrane = HomologicalMembraneChain(n_0cells=16, n_1cells=32, n_2cells=16)
    laplacian = HigherDimensionalLaplacian(membrane.d1, membrane.d2)
    rng = np.random.default_rng(seed=42)
    test_chain = rng.normal(0, 1.0, size=(32,))

    exact, coexact, harmonic = laplacian.hodge_decomposition(test_chain)
    np.testing.assert_allclose(exact + coexact + harmonic, test_chain, atol=1e-9)
    assert abs(np.dot(exact, coexact)) < 1e-8
    assert abs(np.dot(exact, harmonic)) < 1e-8
    assert abs(np.dot(coexact, harmonic)) < 1e-8
    assert laplacian.verify_boundary_orthogonality(test_chain)


def test_betti_number_and_spectral_gap():
    # The boundary of a two-edge path has full column rank and beta_1 = 0.
    d1 = np.array([[1.0, -1.0], [0.0, 1.0], [0.0, 0.0]])
    laplacian = HigherDimensionalLaplacian(d1, np.zeros((2, 0)))
    assert laplacian.dimension == 2
    assert laplacian.compute_betti_number() == 0
    assert laplacian.compute_spectral_gap() > 0

    zero = HigherDimensionalLaplacian(np.zeros((0, 2)), np.zeros((2, 0)))
    assert zero.compute_betti_number() == 2
    assert zero.compute_spectral_gap() == 0.0


def test_laplacian_matches_boundary_formula_and_hodge_components():
    d_k = np.array([[1.0, -1.0, 0.0]])
    d_kp1 = np.zeros((3, 1))
    operator = HigherDimensionalLaplacian(d_k, d_kp1)
    np.testing.assert_allclose(operator.L_k, d_k.T @ d_k + d_kp1 @ d_kp1.T)
    vector = np.array([2.0, 1.0, -3.0])
    exact, coexact, harmonic = operator.hodge_decomposition(vector)
    np.testing.assert_allclose(exact + coexact + harmonic, vector, atol=1e-10)


def test_laplacian_rejects_invalid_operators_and_vectors():
    with pytest.raises(ValueError, match="two-dimensional"):
        HigherDimensionalLaplacian(np.ones(3), np.ones((3, 1)))
    with pytest.raises(ValueError, match="match"):
        HigherDimensionalLaplacian(np.ones((2, 3)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="chain property"):
        HigherDimensionalLaplacian(np.ones((1, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        HigherDimensionalLaplacian(np.array([[np.nan]]), np.zeros((1, 0)))
    with pytest.raises(ValueError, match="chain_tol"):
        HigherDimensionalLaplacian(np.zeros((0, 1)), np.zeros((1, 0)), chain_tol=-1)

    operator = HigherDimensionalLaplacian(np.zeros((0, 2)), np.zeros((2, 0)))
    with pytest.raises(ValueError, match="zero_tol"):
        operator.compute_betti_number(-1)
    with pytest.raises(ValueError, match="zero_tol"):
        operator.compute_spectral_gap(np.nan)
    with pytest.raises(ValueError, match="shape"):
        operator.hodge_decomposition(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        operator.hodge_decomposition(np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="atol"):
        operator.verify_boundary_orthogonality(np.ones(2), atol=-1)


def test_evt_tracks_drift_and_detects_spike_without_contamination():
    rng = np.random.default_rng(seed=42)
    time = np.linspace(0, 20, 1000)
    baseline = 2.0 * np.sin(0.5 * time) + rng.normal(0, 0.5, size=time.size)
    estimator = NonStationaryPOTEstimator(risk_level=1e-3)
    estimator.fit_initial_window(baseline[:300])

    false_positives = 0
    for value in baseline[300:800]:
        is_anomaly, _ = estimator.update_dynamic_threshold(value)
        false_positives += int(is_anomaly)
    prior_size = len(estimator.window)
    is_spike, spike_threshold = estimator.update_dynamic_threshold(15.0)

    assert false_positives <= 3
    assert is_spike is True
    assert spike_threshold < 15.0
    assert len(estimator.window) == prior_size


def test_evt_bound_scales_with_horizon_and_validates_state():
    estimator = NonStationaryPOTEstimator(risk_level=0.001, alpha_decay=0.9)
    estimator.fit_initial_window(np.linspace(-1.0, 1.0, 500))
    one_step = estimator.compute_extreme_value_bound()
    longer_horizon = estimator.compute_extreme_value_bound(horizon=20)
    assert longer_horizon >= one_step
    with pytest.raises(ValueError, match="positive"):
        estimator.compute_extreme_value_bound(horizon=0)
    with pytest.raises(TypeError, match="integer"):
        estimator.compute_extreme_value_bound(horizon=1.5)

    unfit = NonStationaryPOTEstimator()
    with pytest.raises(RuntimeError, match="fitted"):
        unfit.compute_extreme_value_bound()
    with pytest.raises(RuntimeError, match="fitted"):
        unfit.update_dynamic_threshold(1.0)


def test_evt_initialization_validation_and_finite_value_handling():
    for kwargs in (
        {"risk_level": 0.0},
        {"init_quantile": 1.0},
        {"risk_level": 0.05, "init_quantile": 0.98},
        {"alpha_decay": 1.0},
        {"window_size": 0},
        {"refit_interval": 1.5},
        {"min_exceedances": 1},
    ):
        with pytest.raises((ValueError, TypeError)):
            NonStationaryPOTEstimator(**kwargs)

    estimator = NonStationaryPOTEstimator(window_size=20)
    with pytest.raises(ValueError, match="finite value"):
        estimator.fit_initial_window(np.array([np.nan, np.inf]))
    estimator.fit_initial_window(np.array([np.nan, 0.0, 1.0, 2.0, 3.0]))
    assert len(estimator.window) == 4
    previous_window = list(estimator.window)
    is_anomaly, _ = estimator.update_dynamic_threshold(np.nan)
    assert is_anomaly is False
    assert estimator.window == previous_window
    with pytest.raises(ValueError, match="finite scalar"):
        estimator.update_dynamic_threshold("not-a-number")


def test_evt_caps_history_and_supports_zero_decay():
    estimator = NonStationaryPOTEstimator(
        risk_level=0.01,
        init_quantile=0.9,
        alpha_decay=0.0,
        window_size=30,
        refit_interval=2,
    )
    estimator.fit_initial_window(np.linspace(0.0, 1.0, 100))
    assert len(estimator.window) == 30
    assert estimator.t0 == estimator.window[-1]
    for value in np.linspace(0.1, 0.2, 40):
        estimator.update_dynamic_threshold(float(value))
    assert len(estimator.window) <= 30


def test_evt_zero_shape_limit_and_flat_tail_bound():
    estimator = NonStationaryPOTEstimator(risk_level=0.01, init_quantile=0.9)
    estimator.fit_initial_window(np.linspace(0.0, 1.0, 100))
    estimator.gpd_gamma = 0.0
    estimator.gpd_sigma = 2.0
    bound = estimator.compute_extreme_value_bound()
    assert bound > estimator.t0
