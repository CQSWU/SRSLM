"""The retained zero-trace control changes only its intended input."""
import numpy as np
import pytest

from pomapf_env.trace_variant import TraceVariant


def test_zero_trace_changes_only_trace():
    obs = [{"tau": np.ones((1, 11, 11)), "obs": np.ones((3, 15, 15)),
            "tau_free_mask": np.ones((1, 11, 11))}]
    original = {key: value.copy() for key, value in obs[0].items()}
    TraceVariant("zero").apply(obs)
    assert not obs[0]["tau"].any()
    for key in ("obs", "tau_free_mask"):
        np.testing.assert_array_equal(obs[0][key], original[key])


def test_shuffled_trace_and_silent_missing_trace_are_rejected():
    with pytest.raises(ValueError, match="trace variant"):
        TraceVariant("shuffled")
    with pytest.raises(KeyError, match="tau"):
        TraceVariant("zero").apply([{}])
