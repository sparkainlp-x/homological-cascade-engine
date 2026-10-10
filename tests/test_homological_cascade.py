"""Tests for the chain condition and diffusion operators."""

import numpy as np
import pytest

from src.homological_cascade import (
    EventCascadeOperator,
    HomologicalMembraneChain,
    ObserverManifestOperator,
)


def test_boundary_constraint_and_reproducibility():
    chain = HomologicalMembraneChain()
    other = HomologicalMembraneChain()
    assert chain.d1.shape == (32, 64)
    assert chain.d2.shape == (64, 32)
    assert np.linalg.norm(chain.d1 @ chain.d2, ord=2) < 1e-12
    np.testing.assert_array_equal(chain.d1, other.d1)
    np.testing.assert_allclose(chain.d2, other.d2)


def test_exact_diffusion_is_stable_and_respects_steps():
    chain = HomologicalMembraneChain(n_0cells=4, n_1cells=8, n_2cells=2)
    operator = EventCascadeOperator(chain)
    initial = np.arange(1.0, 9.0)
    result = operator.propagate_cascade(initial, steps=5)
    assert result.shape == initial.shape
    assert np.linalg.norm(result) <= np.linalg.norm(initial) + 1e-10
    np.testing.assert_array_equal(operator.propagate_cascade(initial, steps=0), initial)
    assert not np.allclose(result, initial)


def test_observer_manifestation_excludes_nan_and_keeps_high_energy():
    state = np.array([1.0, -2.0, 3.0, np.nan])
    mask, values = ObserverManifestOperator(0.75).collapse_manifestation(state)
    assert mask.tolist() == [0, 0, 1, 0]
    np.testing.assert_allclose(values, [0.0, 0.0, 3.0, 0.0])


def test_chain_and_operator_validation():
    with pytest.raises(ValueError):
        HomologicalMembraneChain(n_0cells=0)
    with pytest.raises(TypeError):
        HomologicalMembraneChain(n_1cells=2.5)
    with pytest.raises(ValueError):
        EventCascadeOperator(HomologicalMembraneChain(), coupling_strength=-1)
    with pytest.raises(ValueError):
        EventCascadeOperator(HomologicalMembraneChain(n_1cells=4)).propagate_cascade(np.ones(3))
    with pytest.raises(ValueError):
        ObserverManifestOperator(1.1)
    with pytest.raises(ValueError):
        ObserverManifestOperator().collapse_manifestation(np.array([np.nan]))


def test_kernel_boundary_padding_and_invalid_shapes():
    padded = HomologicalMembraneChain._build_kernel_boundary(np.eye(2), 3)
    assert padded.shape == (2, 3)
    assert np.count_nonzero(padded) == 0
    with pytest.raises(ValueError):
        HomologicalMembraneChain._build_kernel_boundary(np.ones(3), 1)
    with pytest.raises(TypeError):
        HomologicalMembraneChain._build_kernel_boundary(np.eye(2), True)
    with pytest.raises(ValueError):
        HomologicalMembraneChain._build_kernel_boundary(np.eye(2), -1)
    with pytest.raises(TypeError):
        HomologicalMembraneChain(seed=1.5)


def test_diffusion_validation_and_observer_empty_state():
    operator = EventCascadeOperator(HomologicalMembraneChain(n_0cells=3, n_1cells=5, n_2cells=1))
    with pytest.raises(ValueError):
        operator.propagate_cascade(np.ones(5), steps=-1)
    with pytest.raises(ValueError):
        operator.propagate_cascade(np.ones(5), dt=-0.1)
    with pytest.raises(ValueError):
        operator.propagate_cascade(np.full(5, np.nan))
    with pytest.raises(ValueError):
        ObserverManifestOperator().collapse_manifestation(np.array([]))
