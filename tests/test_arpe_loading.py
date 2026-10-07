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
        ("action_sampling", "torch"),
    ],
)
def test_retired_inference_overrides_are_not_accepted(field, value):
    with pytest.raises(ValidationError, match=field):
        EPOMTraceContextConfig(
            path_to_weights="unused", milestone_checkpoint="missing.pth", **{field: value}
        )


def test_checkpoint_selection_never_silently_falls_back_to_best(tmp_path):
    (tmp_path / "best_0001.pth").touch()
    backbone = PolicyBackbone.__new__(PolicyBackbone)
    backbone.algo_cfg = PolicyBackboneConfig(
        path_to_weights=str(tmp_path),
        milestone_checkpoint=str(tmp_path / "missing.pth"),
    )
    with pytest.raises(FileNotFoundError):
        backbone._load_checkpoint(torch.device("cpu"))


def test_checkpoint_path_is_required():
    with pytest.raises(ValidationError, match="milestone_checkpoint"):
        EPOMTraceContextConfig(path_to_weights="unused")


def test_an_explicit_checkpoint_can_be_stored_outside_its_config_directory(tmp_path):
    current_dir = tmp_path / "run" / "checkpoint_p0"
    candidate = tmp_path / "downloaded" / "checkpoint.pth"
    candidate.parent.mkdir(parents=True)
    torch.save({"model": {"weight": torch.ones(2)}}, candidate)
    adapter = EPOMTraceContext.__new__(EPOMTraceContext)
    adapter.algo_cfg = EPOMTraceContextConfig(
        path_to_weights=str(current_dir.parent),
        milestone_checkpoint=str(candidate),
    )
    loaded = adapter._load_checkpoint(torch.device("cpu"))
    torch.testing.assert_close(loaded["model"]["weight"], torch.ones(2))
    assert adapter.checkpoint_path == candidate.resolve()
