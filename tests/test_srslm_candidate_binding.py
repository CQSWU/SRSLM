from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agents.srslm import SRSLM, SRSLMConfig
from agents.switcher import SwitcherConfig


class _FrozenPolicy:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()))

    def parameters(self):
        return [self.weight]


class _FakeCandidate:
    def __init__(self, _artifact, *, seed, device):
        self.seed = seed
        self.device = torch.device("cpu")
        self.ppo = _FrozenPolicy()

    def after_reset(self):
        pass

    def verify_frozen(self):
        return {"verified": True}


class _FakeSwitcher:
    candidate_artifact = SimpleNamespace(
        project_root=None,
    )

    def __init__(self, _cfg):
        self.candidate_artifact = SimpleNamespace(
            project_root=Path(__file__).resolve().parents[1],
        )


class _MissingCandidateSwitcher(_FakeSwitcher):
    def __init__(self, _cfg):
        self.candidate_artifact = None


class _FakePlanner:
    def __init__(self, **_kwargs):
        pass

    def reset(self):
        pass


def _config():
    return SRSLMConfig(
        switcher=SwitcherConfig(path_to_weights="unused"),
    )


def test_srslm_accepts_the_caar_artifact_pinned_by_switcher():
    algorithm = SRSLM(
        _config(),
        candidate_factory=_FakeCandidate,
        switcher_factory=_FakeSwitcher,
        planner_factory=_FakePlanner,
    )

    assert algorithm.candidate.seed == 0
    assert algorithm.candidate.device == torch.device("cpu")


def test_srslm_rejects_a_switcher_without_a_pinned_candidate():
    with pytest.raises(RuntimeError, match="does not pin"):
        SRSLM(
            _config(),
            candidate_factory=_FakeCandidate,
            switcher_factory=_MissingCandidateSwitcher,
            planner_factory=_FakePlanner,
        )
