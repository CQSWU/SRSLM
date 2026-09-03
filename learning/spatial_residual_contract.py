"""Strict, mask-free input contracts for the v4 spatial residual actor.

This module is intentionally Torch-free.  Training, checkpoint selection, and
independent evaluation can therefore resolve the same dimensions and parameter
budgets without constructing a model or loading a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SPATIAL_RESIDUAL_ARCHITECTURES = (
    "tiny_residual16",
    "linear_spatial_residual",
)
SPATIAL_RESIDUAL_INPUT_CONTRACTS = (
    "P+q+H",
    "P+q",
    "P+H",
    "P-only",
)
SPATIAL_RESIDUAL_TRACE_VIEWS = (
    "P121",
    "center-P5",
)

_ARCHITECTURE_HIDDEN_DIMS = {
    "tiny_residual16": 16,
    "linear_spatial_residual": 0,
}
_INPUT_ORDER = {
    "P+q+H": ("P", "q", "H"),
    "P+q": ("P", "q"),
    "P+H": ("P", "H"),
    "P-only": ("P",),
}
_TRACE_FEATURES = {
    "P121": 121,
    "center-P5": 5,
}
_TRACE_INPUT_NAMES = {
    "P121": "trace_11x11_flatten_row_major_121",
    "center-P5": "trace_center_stay_up_down_left_right_5",
}
_TRACE_ORDERS = {
    "P121": "row_major_y_then_x",
    "center-P5": "stay_up_down_left_right",
}
_COMPONENT_INPUT_NAMES = {
    "q": "base_policy_probabilities_5",
    "H": "base_policy_entropy_1",
}
_COMPONENT_DIMENSIONS = {
    "q": 5,
    "H": 1,
}


@dataclass(frozen=True)
class SpatialResidualContract:
    """Resolved architecture contract for one checkpoint family."""

    architecture: str
    input_contract: str
    hidden_dim: int
    trace_view: str

    @property
    def input_order(self) -> tuple[str, ...]:
        return _INPUT_ORDER[self.input_contract]

    @property
    def trace_features(self) -> int:
        return _TRACE_FEATURES[self.trace_view]

    @property
    def input_dimensions(self) -> dict[str, int]:
        dimensions = {"P": self.trace_features}
        dimensions.update(
            (name, _COMPONENT_DIMENSIONS[name])
            for name in self.input_order
            if name != "P"
        )
        return dimensions

    @property
    def actor_inputs(self) -> list[str]:
        names = {"P": _TRACE_INPUT_NAMES[self.trace_view], **_COMPONENT_INPUT_NAMES}
        return [names[name] for name in self.input_order]

    @property
    def actor_input_features(self) -> int:
        dimensions = self.input_dimensions
        return sum(dimensions[name] for name in self.input_order)

    @property
    def actor_trainable_parameters(self) -> int:
        width = self.actor_input_features
        if self.hidden_dim == 0:
            return width * 5 + 5
        return width * self.hidden_dim + self.hidden_dim + self.hidden_dim * 5 + 5

    @property
    def total_trainable_parameters(self) -> int:
        # The critic branch is invariant across all v4 actor ablations.
        return 27_761 + self.actor_trainable_parameters

    @property
    def uses_probabilities(self) -> bool:
        return "q" in self.input_order

    @property
    def uses_entropy(self) -> bool:
        return "H" in self.input_order

    @property
    def trace_order(self) -> str:
        return _TRACE_ORDERS[self.trace_view]

    @property
    def provenance_name(self) -> str:
        # Preserve the two already-frozen v4 names exactly.
        if self.input_contract == "P+q+H" and self.trace_view == "P121":
            return {
                "tiny_residual16": (
                    "tiny_residual16_unmasked_spatial_p_probs_entropy_v4"
                ),
                "linear_spatial_residual": (
                    "linear_spatial_residual_unmasked_spatial_p_probs_entropy_v4"
                ),
            }[self.architecture]
        architecture = {
            "tiny_residual16": "tiny_residual16",
            "linear_spatial_residual": "linear_spatial_residual",
        }[self.architecture]
        view = {"P121": "p121", "center-P5": "center_p5"}[self.trace_view]
        inputs = {
            "P+q+H": "p_q_h",
            "P+q": "p_q",
            "P+H": "p_h",
            "P-only": "p_only",
        }[self.input_contract]
        return f"{architecture}_unmasked_{view}_{inputs}_v4"

    def checkpoint_fields(self) -> dict[str, object]:
        """Canonical provenance fields consumed by loaders and auditors."""

        return {
            "spatial_residual_input_contract": self.input_contract,
            "spatial_residual_input_order": list(self.input_order),
            "spatial_residual_input_dimensions": self.input_dimensions,
            "spatial_residual_trace_view": self.trace_view,
            "actor_inputs": self.actor_inputs,
            "actor_input_features": self.actor_input_features,
            "actor_hidden_features": self.hidden_dim,
            "actor_trainable_parameters": self.actor_trainable_parameters,
            "trainable_parameters": self.total_trainable_parameters,
        }


def resolve_spatial_residual_contract(
    architecture: str,
    settings: Mapping[str, object] | None = None,
    *,
    input_contract: str | None = None,
    hidden_dim: int | None = None,
    trace_view: str | None = None,
) -> SpatialResidualContract:
    """Resolve defaults and reject architecture-changing ablation drift."""

    if architecture not in _ARCHITECTURE_HIDDEN_DIMS:
        raise ValueError(f"Not a v4 spatial residual architecture: {architecture!r}.")
    settings = settings or {}
    resolved_input = input_contract or str(
        settings.get("trace_spatial_input_contract", "P+q+H")
    )
    resolved_view = trace_view or str(
        settings.get("trace_spatial_trace_view", "P121")
    )
    configured_hidden = (
        hidden_dim
        if hidden_dim is not None
        else settings.get("trace_spatial_hidden_dim")
    )
    expected_hidden = _ARCHITECTURE_HIDDEN_DIMS[architecture]
    resolved_hidden = (
        expected_hidden if configured_hidden is None else int(configured_hidden)
    )

    if resolved_input not in _INPUT_ORDER:
        raise ValueError(
            "trace_spatial_input_contract must be one of "
            f"{SPATIAL_RESIDUAL_INPUT_CONTRACTS}, got {resolved_input!r}."
        )
    if resolved_view not in _TRACE_FEATURES:
        raise ValueError(
            "trace_spatial_trace_view must be one of "
            f"{SPATIAL_RESIDUAL_TRACE_VIEWS}, got {resolved_view!r}."
        )
    if resolved_hidden != expected_hidden:
        raise ValueError(
            f"Architecture {architecture!r} fixes trace_spatial_hidden_dim="
            f"{expected_hidden}, got {resolved_hidden}."
        )
    return SpatialResidualContract(
        architecture=architecture,
        input_contract=resolved_input,
        hidden_dim=resolved_hidden,
        trace_view=resolved_view,
    )


__all__ = [
    "SPATIAL_RESIDUAL_ARCHITECTURES",
    "SPATIAL_RESIDUAL_INPUT_CONTRACTS",
    "SPATIAL_RESIDUAL_TRACE_VIEWS",
    "SpatialResidualContract",
    "resolve_spatial_residual_contract",
]
