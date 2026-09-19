"""Keep retired inference variants out of the current, portable loader."""

import pytest
import torch
from pydantic import ValidationError

from agents.epom_trace_context import (
    EPOMTraceContext,
    EPOMTraceContextConfig,
    _validate_r5_trace_contract,
)
from agents.policy_backbone import PolicyBackbone, PolicyBackboneConfig


@pytest.mark.parametrize("field,value", [
    ("learned_gate_override", "all"),
    ("entropy_threshold_override", 0.9),
    ("checkpoint_kind", "auto"),
])
def test_retired_inference_overrides_are_not_accepted(field, value):
    with pytest.raises(ValidationError):
        EPOMTraceContextConfig(path_to_weights="unused", **{field: value})


def test_checkpoint_selection_never_silently_falls_back_to_best(tmp_path):
    (tmp_path / "best_0001.pth").touch()
    backbone = PolicyBackbone.__new__(PolicyBackbone)
    with pytest.raises(ValidationError):
        PolicyBackboneConfig(path_to_weights=str(tmp_path), checkpoint_kind="auto")
    with pytest.raises(ValueError, match="explicit checkpoint kind"):
        backbone._load_checkpoint(tmp_path, torch.device("cpu"), "auto")
    with pytest.raises(FileNotFoundError):
        backbone._load_checkpoint(tmp_path, torch.device("cpu"), "latest")


@pytest.mark.parametrize("settings", [
    {"checkpoint_kind": "milestone"},
    {"checkpoint_kind": "milestone", "milestone_checkpoint": " "},
    {"checkpoint_kind": "latest", "milestone_checkpoint": "ignored.pth"},
])
def test_milestone_selection_cannot_be_missing_or_ignored(settings):
    with pytest.raises(ValidationError):
        EPOMTraceContextConfig(path_to_weights="unused", **settings)


def test_an_explicit_checkpoint_can_be_stored_outside_its_config_directory(tmp_path):
    current_dir = tmp_path / "run" / "checkpoint_p0"
    candidate = tmp_path / "downloaded" / "checkpoint.pth"
    candidate.parent.mkdir(parents=True)
    torch.save({"model": {"weight": torch.ones(2)}}, candidate)
    adapter = EPOMTraceContext.__new__(EPOMTraceContext)
    adapter.algo_cfg = EPOMTraceContextConfig(
        path_to_weights=str(current_dir.parent),
        checkpoint_kind="milestone",
        milestone_checkpoint=str(candidate),
    )
    loaded = adapter._load_checkpoint(current_dir, torch.device("cpu"), "milestone")
    torch.testing.assert_close(loaded["model"]["weight"], torch.ones(2))
    assert adapter.checkpoint_path == candidate.resolve()


def _trace_config():
    return {
        "experiment_settings": {"trace_context_architecture": "paper_entropy_fusion"},
        "environment": {"tau_radius": 5, "tau_raw": False, "trace_variant": "real"},
    }


@pytest.mark.parametrize("architecture", ["context", "paper_entropy_multiplier", None])
def test_trace_contract_does_not_infer_a_legacy_architecture(architecture):
    config = _trace_config()
    if architecture is None:
        del config["experiment_settings"]["trace_context_architecture"]
    else:
        config["experiment_settings"]["trace_context_architecture"] = architecture
    with pytest.raises(RuntimeError):
        _validate_r5_trace_contract(config)


def test_trace_contract_preserves_centered_real_and_zero_controls():
    config = _trace_config()
    for variant in ("real", "zero"):
        config["environment"]["trace_variant"] = variant
        contract = _validate_r5_trace_contract(config)
        assert contract["tau_raw"] is False
        assert contract["tau_size"] == 11
        assert contract["trace_variant"] == variant


def test_historical_action_sampling_is_explicit_and_unchanged():
    assert EPOMTraceContextConfig(path_to_weights="unused").action_sampling == "torch"
    assert EPOMTraceContextConfig(
        path_to_weights="unused", action_sampling="direct_numpy"
    ).action_sampling == "direct_numpy"
