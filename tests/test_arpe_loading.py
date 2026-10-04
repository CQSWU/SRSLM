import pytest
import torch
from pydantic import ValidationError

from agents.epom_trace_context import (
    EPOMTraceContext,
    EPOMTraceContextConfig,
)
from agents.policy_backbone import PolicyBackbone, PolicyBackboneConfig


@pytest.mark.parametrize(
    "field,value",
    [
        ("learned_gate_override", "all"),
        ("entropy_threshold_override", 0.9),
        ("checkpoint_kind", "auto"),
    ],
)
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


@pytest.mark.parametrize(
    "settings",
    [
        {"checkpoint_kind": "milestone"},
        {"checkpoint_kind": "milestone", "milestone_checkpoint": " "},
        {"checkpoint_kind": "latest", "milestone_checkpoint": "ignored.pth"},
    ],
)
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


def test_historical_action_sampling_is_explicit_and_unchanged():
    assert EPOMTraceContextConfig(path_to_weights="unused").action_sampling == "torch"
    assert (
        EPOMTraceContextConfig(
            path_to_weights="unused", action_sampling="direct_numpy"
        ).action_sampling
        == "direct_numpy"
    )
