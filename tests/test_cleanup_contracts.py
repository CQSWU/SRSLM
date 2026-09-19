"""Retired experiment branches and mixed-source result journals stay closed."""
import json

import numpy as np
import pytest

import run_experiments as runner
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


def test_journal_cannot_resume_after_code_change(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    task = {"algorithm": "ARPE", "map_name": "m", "num_agents": 100, "seed": 0}
    monkeypatch.setattr(runner, "evaluation_source_sha256", lambda: "1" * 64)
    runner._initialize_result_journal(path, "a" * 64, 1)
    assert runner._load_result_journal(path, "a" * 64, [task]) == []
    original = path.read_bytes()
    monkeypatch.setattr(runner, "evaluation_source_sha256", lambda: "2" * 64)
    with pytest.raises(ValueError, match="old implementation"):
        runner._load_result_journal(path, "a" * 64, [task])
    assert path.read_bytes() == original

    path.write_bytes(original + b'{"truncated":')
    damaged = path.read_bytes()
    with pytest.raises(ValueError, match="old implementation"):
        runner._load_result_journal(path, "a" * 64, [task], repair_final_record=True)
    assert path.read_bytes() == damaged


def test_old_journal_without_source_identity_is_not_resumed(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"record_type": "header",
        "schema": "experiment_result_journal_v1", "contract_sha256": "a" * 64,
        "expected_tasks": 0}) + "\n")
    with pytest.raises(ValueError, match="old implementation"):
        runner._load_result_journal(path, "a" * 64, [])


def test_fingerprint_covers_model_config_and_maps_not_cache(tmp_path):
    for name in ("agents", "learning", "maps"):
        (tmp_path / name).mkdir()
    (tmp_path / "agents/model.py").write_text("version = 2\n")
    first = runner.evaluation_source_sha256(tmp_path)
    (tmp_path / "agents/__pycache__").mkdir()
    (tmp_path / "agents/__pycache__/model.pyc").write_bytes(b"cache")
    assert runner.evaluation_source_sha256(tmp_path) == first
    (tmp_path / "maps/test.yaml").write_text("m: .\n")
    second = runner.evaluation_source_sha256(tmp_path)
    assert second != first
    (tmp_path / "agents/model.py").write_text("version = 1\n")
    assert runner.evaluation_source_sha256(tmp_path) != second
