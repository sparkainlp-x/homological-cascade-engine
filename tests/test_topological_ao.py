"""Tests for the classical, synthetic co-exact residual-energy scorer.

All inputs are SYNTHETIC. The "localized offset" frames below are a stand-in
for a tamper scenario in a toy setting; they are not models of a real
eavesdropping attack, and passing these tests says nothing about QKD security.
"""

import numpy as np
import pytest

from src.analog_calibration import AnalogSoundCalibrator
from src.quantum import TopologicalAdaptiveOpticsFilter


def _spec_stream(seed: int = 42, n_samples: int = 500, n_channels: int = 256):
    rng = np.random.default_rng(seed=seed)
    time = np.linspace(0, 5, n_samples)
    drift = 2.0 * np.sin(0.4 * time)[:, None]
    return drift + rng.normal(0, 0.5, size=(n_samples, n_channels))


@pytest.fixture(scope="module")
def spec_stream():
    return _spec_stream()


def test_spec_scenario_drift_and_synthetic_localized_offset(spec_stream):
    """Specification scenario, with one assertion corrected.

    The specification compared ``var(clean_phase)`` against the variance of
    the *raw* frame. That does not hold: the calibrator divides by a robust
    channel scale (about 0.67 here), so calibrated frames have larger variance
    than raw frames. The true property is that diffusion does not increase the
    variance of the vector it is applied to (the conditioned frame).
    """
    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=256)
    ao_filter.calibrate_ambient_sky(spec_stream[:200])

    false_alarms = 0
    for i in range(200, 400):
        frame = spec_stream[i]
        conditioned = ao_filter.condition_frame(frame)
        flagged, clean_phase, _ = ao_filter.process_photon_frame(frame)
        false_alarms += int(flagged)
        assert np.var(clean_phase) <= np.var(conditioned) + 1e-5
    assert false_alarms <= 2

    # Synthetic localized offset (stand-in for a tamper scenario).
    offset_frame = spec_stream[450].copy()
    offset_frame[50:100] += 18.0
    flagged, _, energy = ao_filter.process_photon_frame(offset_frame)
    assert flagged is True
    assert energy > 10 * ao_filter.pot_estimator.compute_extreme_value_bound()


def test_uncalibrated_filter_raises():
    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=64)
    with pytest.raises(RuntimeError):
        ao_filter.process_photon_frame(np.zeros(64))
    with pytest.raises(RuntimeError):
        ao_filter.condition_frame(np.zeros(64))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_subapertures": 1}, ValueError),
        ({"n_subapertures": 3.5}, TypeError),
        ({"n_subapertures": True}, TypeError),
        ({"coupling_j": -1.0}, ValueError),
        ({"coupling_j": "x"}, ValueError),
        ({"dt": float("nan")}, ValueError),
        ({"sample_rate_hz": 0.0}, ValueError),
        ({"risk_level": 0.5}, ValueError),
    ],
)
def test_constructor_validation(kwargs, error):
    with pytest.raises(error):
        TopologicalAdaptiveOpticsFilter(**kwargs)


def test_shape_and_finiteness_validation():
    rng = np.random.default_rng(1)
    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=64)
    with pytest.raises(ValueError):
        ao_filter.calibrate_ambient_sky(rng.normal(size=(50, 63)))
    with pytest.raises(ValueError):
        ao_filter.calibrate_ambient_sky(rng.normal(size=(1, 64)))
    ao_filter.calibrate_ambient_sky(rng.normal(size=(50, 64)))
    with pytest.raises(ValueError):
        ao_filter.process_photon_frame(np.zeros(65))
    with pytest.raises(ValueError):
        ao_filter.process_photon_frame(np.zeros((2, 64)))
    bad = np.zeros(64)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        ao_filter.process_photon_frame(bad)


def test_deterministic_under_seed(spec_stream):
    outputs = []
    for _ in range(2):
        ao_filter = TopologicalAdaptiveOpticsFilter()
        ao_filter.calibrate_ambient_sky(spec_stream[:200])
        outputs.append([ao_filter.process_photon_frame(f) for f in spec_stream[200:230]])
    for (f1, r1, e1), (f2, r2, e2) in zip(*outputs):
        assert f1 == f2 and e1 == e2
        np.testing.assert_array_equal(r1, r2)


def test_diffusion_operator_is_symmetric_contraction():
    ao_filter = TopologicalAdaptiveOpticsFilter()
    operator = np.asarray(ao_filter.U_diffusion)
    np.testing.assert_allclose(operator, operator.T, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(operator)
    assert eigenvalues.min() > 0
    assert eigenvalues.max() <= 1 + 1e-12
    assert np.linalg.norm(operator, ord=2) <= 1 + 1e-12
    vector = np.random.default_rng(3).normal(size=256)
    assert np.linalg.norm(operator @ vector) <= np.linalg.norm(vector) + 1e-12
    # Harmonic vectors (L1 h = 0) are left unchanged.
    _, _, harmonic = ao_filter.laplacian_solver.hodge_decomposition(vector)
    np.testing.assert_allclose(operator @ harmonic, harmonic, atol=1e-10)


def test_reconstruction_and_energy_match_definitions(spec_stream):
    ao_filter = TopologicalAdaptiveOpticsFilter()
    ao_filter.calibrate_ambient_sky(spec_stream[:200])
    frame = spec_stream[300]
    _, reconstructed, energy = ao_filter.process_photon_frame(frame)
    conditioned = ao_filter.condition_frame(frame)
    assert abs(conditioned.mean()) < 1e-12
    np.testing.assert_allclose(reconstructed, ao_filter.U_diffusion @ conditioned)
    _, coexact, _ = ao_filter.laplacian_solver.hodge_decomposition(reconstructed)
    assert energy == pytest.approx(float(np.linalg.norm(coexact) ** 2))


def test_pure_noise_false_alarm_count_is_bounded():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1.0, size=(700, 128))
    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=128)
    ao_filter.calibrate_ambient_sky(noise[:200])
    alarms = sum(ao_filter.process_photon_frame(frame)[0] for frame in noise[200:])
    assert alarms <= 3


def test_piston_removal_cancels_common_mode_offset():
    rng = np.random.default_rng(11)
    baseline = rng.normal(size=(100, 64))
    with_piston = TopologicalAdaptiveOpticsFilter(n_subapertures=64)
    without_piston = TopologicalAdaptiveOpticsFilter(n_subapertures=64, remove_piston=False)
    with_piston.calibrate_ambient_sky(baseline)
    without_piston.calibrate_ambient_sky(baseline)
    # A frame equal to the calibrated centre plus a constant in calibrated units.
    frame = with_piston.calibrator.center + 5.0 * with_piston.calibrator.scale
    assert with_piston.process_photon_frame(frame)[2] == pytest.approx(0.0, abs=1e-18)
    assert without_piston.process_photon_frame(frame)[2] > 1.0


def test_flag_is_energy_based_not_tamper_specific(spec_stream):
    """Any large broadband disturbance is flagged, not only localized offsets."""
    ao_filter = TopologicalAdaptiveOpticsFilter()
    ao_filter.calibrate_ambient_sky(spec_stream[:200])
    burst = spec_stream[210] + np.random.default_rng(5).normal(0, 10.0, size=256)
    assert ao_filter.process_photon_frame(burst)[0] is True
    # A small localized offset stays below the bound.
    small = spec_stream[211].copy()
    small[50:100] += 0.05
    assert ao_filter.process_photon_frame(small)[0] is False


def test_calibrator_robust_statistics_and_fallbacks():
    calibrator = AnalogSoundCalibrator(sample_rate_hz=1000.0, n_channels=3)
    assert not calibrator.is_calibrated
    block = np.array(
        [
            [1.0, 5.0, np.nan],
            [2.0, 5.0, np.nan],
            [3.0, 5.0, np.nan],
            [100.0, 5.0, np.nan],
        ]
    )
    calibrator.calibrate_baseline(block)
    assert calibrator.center[0] == pytest.approx(2.5)
    assert calibrator.scale[0] == pytest.approx(1.4826)
    # Constant channel: MAD and std are zero, so the scale falls back to 1.
    assert calibrator.center[1] == 5.0 and calibrator.scale[1] == 1.0
    # All-NaN channel keeps centre 0 and scale 1.
    assert calibrator.center[2] == 0.0 and calibrator.scale[2] == 1.0
    np.testing.assert_allclose(
        calibrator.process_frame([2.5, 6.0, 2.0]), [0.0, 1.0, 2.0]
    )
    np.testing.assert_allclose(
        calibrator.process_block_whitened([[2.5, 6.0, 2.0]]), [[0.0, 1.0, 2.0]]
    )


def test_calibrator_mad_zero_falls_back_to_std():
    calibrator = AnalogSoundCalibrator(n_channels=1)
    calibrator.calibrate_baseline(np.array([[0.0], [0.0], [0.0], [4.0]]))
    assert calibrator.scale[0] == pytest.approx(np.std([0.0, 0.0, 0.0, 4.0]))


def test_calibrator_validation():
    with pytest.raises(ValueError):
        AnalogSoundCalibrator(sample_rate_hz=float("inf"))
    with pytest.raises(ValueError):
        AnalogSoundCalibrator(sample_rate_hz="fast")
    with pytest.raises(TypeError):
        AnalogSoundCalibrator(n_channels=2.0)
    with pytest.raises(ValueError):
        AnalogSoundCalibrator(n_channels=0)
    calibrator = AnalogSoundCalibrator(n_channels=2)
    with pytest.raises(RuntimeError):
        calibrator.process_frame([0.0, 0.0])
    with pytest.raises(RuntimeError):
        calibrator.process_block_whitened([[0.0, 0.0]])
    with pytest.raises(ValueError):
        calibrator.calibrate_baseline(np.full((3, 2), np.nan))
    with pytest.raises(ValueError):
        calibrator.calibrate_baseline(np.zeros((3, 2), dtype=complex))
    with pytest.raises(ValueError):
        calibrator.calibrate_baseline([["a", "b"], ["c", "d"]])
    calibrator.calibrate_baseline(np.arange(6.0).reshape(3, 2))
    with pytest.raises(ValueError):
        calibrator.process_frame(np.zeros(2, dtype=complex))
    with pytest.raises(ValueError):
        calibrator.process_block_whitened([[np.inf, 0.0]])


def test_hodge_bases_are_cached_per_tolerance():
    ao_filter = TopologicalAdaptiveOpticsFilter(n_subapertures=64)
    solver = ao_filter.laplacian_solver
    vector = np.random.default_rng(2).normal(size=64)
    first = solver.hodge_decomposition(vector)
    second = solver.hodge_decomposition(vector)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert len(solver._basis_cache) == 1
    solver.hodge_decomposition(vector, zero_tol=1e-6)
    assert len(solver._basis_cache) == 2
    np.testing.assert_allclose(sum(first), vector, atol=1e-10)
