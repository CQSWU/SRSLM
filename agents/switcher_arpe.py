"""Checkpoint loader for a Switcher trained with the selected ARPE branch."""

from __future__ import annotations

from typing import Literal

from pydantic import Extra

from agents.switcher import Switcher, SwitcherConfig


ARPE_SWITCHER_LOADER_SCHEMA = "switcher_caar_checkpoint_loader_v1"


class AllStateArpeSwitcherConfig(SwitcherConfig, extra=Extra.forbid):
    name: Literal["AllStateArpeSwitcher"] = "AllStateArpeSwitcher"
    path_to_weights: str
    checkpoint_kind: Literal["latest"] = "latest"


class AllStateArpeSwitcher(Switcher):
    """Load a feed-forward all-state Switcher and reproduce its ARPE pin."""

    expected_encoder_custom = "switcher_all_state"
    allow_aoreplan_wait = True
    policy_label = "all-state ARPE Switcher"

    require_candidate_policy = True

    def __init__(self, cfg: AllStateArpeSwitcherConfig):
        super().__init__(cfg)
        self.loader_schema = ARPE_SWITCHER_LOADER_SCHEMA

    def get_stats(self) -> dict:
        result = super().get_stats()
        result["switcher_loader_schema"] = self.loader_schema
        return result


__all__ = [
    "AllStateArpeSwitcher",
    "AllStateArpeSwitcherConfig",
    "ARPE_SWITCHER_LOADER_SCHEMA",
]
