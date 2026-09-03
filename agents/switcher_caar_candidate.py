"""Strict frozen-CAAR adapter shared by Switcher training and evaluation.

The Switcher must train against the exact CAAR policy that will later be
evaluated.  This module keeps that identity in a small explicit declaration:
the learned CAAR checkpoint, its saved config, and the frozen EPOM-L base
artifact are all addressed by relative paths and SHA256 digests.  The paths
are configurable so a selected training milestone can be pinned without
renaming the network or keeping a version-specific adapter.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from pydantic import Extra

from agents.epom_trace_context import EPOMTraceContext, EPOMTraceContextConfig
from agents.utils_agents import AlgoBase


CAAR_CANDIDATE_KIND = "epom_trace_context_caar_milestone"
CAAR_CANDIDATE_LABEL = "CAAR"
CAAR_CANDIDATE_SCHEMA = "switcher_candidate_caar_v1"
CAAR_TRACE_ARCHITECTURE = (
    "paper_entropy_conv_direct_correction_centered_P_h_z_v3"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(project_root: Path, value: object, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CAAR candidate {field} must be a non-empty path string.")
    declared = Path(value)
    if declared.is_absolute():
        raise ValueError(f"CAAR candidate {field} must be relative to the project.")
    root = Path(project_root).resolve()
    resolved = (root / declared).resolve()
    weights_root = (root / "weights").resolve()
    if resolved != weights_root and weights_root not in resolved.parents:
        raise ValueError(f"CAAR candidate {field} must remain below project/weights.")
    return declared.as_posix(), resolved


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"CAAR candidate {field} must be a SHA256 digest.")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"CAAR candidate {field} is not hexadecimal.")
    return normalized


@dataclass(frozen=True)
class CaarCandidateArtifact:
    """Immutable identity of the CAAR policy used as Switcher branch zero."""

    project_root: Path
    weights_relative: str
    checkpoint_relative: str
    base_weights_relative: str
    base_checkpoint_relative: str
    weights_path: Path
    config_path: Path
    checkpoint_path: Path
    base_weights_path: Path
    base_config_path: Path
    base_checkpoint_path: Path
    checkpoint_sha256: str
    config_sha256: str
    base_checkpoint_sha256: str
    base_config_sha256: str

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        project_root: Path,
    ) -> "CaarCandidateArtifact":
        required = {
            "kind",
            "schema",
            "weights_path",
            "checkpoint_path",
            "checkpoint_sha256",
            "config_sha256",
            "base_weights_path",
            "base_checkpoint_path",
            "base_checkpoint_sha256",
            "base_config_sha256",
            "checkpoint_selection",
            "frozen",
        }
        unknown = set(mapping) - required
        missing = required - set(mapping)
        if unknown or missing:
            raise ValueError(
                "CAAR candidate declaration has unexpected fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        expected_scalars = {
            "kind": CAAR_CANDIDATE_KIND,
            "schema": CAAR_CANDIDATE_SCHEMA,
            "checkpoint_selection": "exact_milestone",
            "frozen": True,
        }
        mismatched = {
            key: {"expected": expected, "actual": mapping.get(key)}
            for key, expected in expected_scalars.items()
            if mapping.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"CAAR candidate declaration is invalid: {mismatched}")

        root = Path(project_root).resolve()
        weights_relative, weights_path = _relative_artifact_path(
            root, mapping["weights_path"], "weights_path"
        )
        checkpoint_relative, checkpoint_path = _relative_artifact_path(
            root, mapping["checkpoint_path"], "checkpoint_path"
        )
        base_weights_relative, base_weights_path = _relative_artifact_path(
            root, mapping["base_weights_path"], "base_weights_path"
        )
        base_checkpoint_relative, base_checkpoint_path = _relative_artifact_path(
            root, mapping["base_checkpoint_path"], "base_checkpoint_path"
        )
        if weights_path not in checkpoint_path.parents:
            raise ValueError("CAAR checkpoint_path must be inside weights_path.")
        if base_weights_path not in base_checkpoint_path.parents:
            raise ValueError(
                "CAAR base_checkpoint_path must be inside base_weights_path."
            )
        return cls(
            project_root=root,
            weights_relative=weights_relative,
            checkpoint_relative=checkpoint_relative,
            base_weights_relative=base_weights_relative,
            base_checkpoint_relative=base_checkpoint_relative,
            weights_path=weights_path,
            config_path=(weights_path / "config.json").resolve(),
            checkpoint_path=checkpoint_path,
            base_weights_path=base_weights_path,
            base_config_path=(base_weights_path / "config.json").resolve(),
            base_checkpoint_path=base_checkpoint_path,
            checkpoint_sha256=_digest(
                mapping["checkpoint_sha256"], "checkpoint_sha256"
            ),
            config_sha256=_digest(mapping["config_sha256"], "config_sha256"),
            base_checkpoint_sha256=_digest(
                mapping["base_checkpoint_sha256"], "base_checkpoint_sha256"
            ),
            base_config_sha256=_digest(
                mapping["base_config_sha256"], "base_config_sha256"
            ),
        )

    @classmethod
    def from_config(
        cls,
        config: "CaarSwitcherCandidateConfig",
        project_root: Path,
    ) -> "CaarCandidateArtifact":
        return cls.from_mapping(config.as_mapping(), project_root)

    def verify_files(self) -> dict[str, str]:
        expected = {
            self.config_path: self.config_sha256,
            self.checkpoint_path: self.checkpoint_sha256,
            self.base_config_path: self.base_config_sha256,
            self.base_checkpoint_path: self.base_checkpoint_sha256,
        }
        verified: dict[str, str] = {}
        for path, digest in expected.items():
            if not path.is_file():
                raise FileNotFoundError(f"Frozen CAAR input is missing: {path}")
            actual = _sha256(path)
            if actual != digest:
                raise RuntimeError(
                    f"Frozen CAAR input changed: {path}: expected {digest}, got {actual}"
                )
            verified[str(path)] = actual
        return verified

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": CAAR_CANDIDATE_KIND,
            "label": CAAR_CANDIDATE_LABEL,
            "schema": CAAR_CANDIDATE_SCHEMA,
            "weights_path": self.weights_relative,
            "config_path": str(self.config_path),
            "checkpoint_path": self.checkpoint_relative,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "base_weights_path": self.base_weights_relative,
            "base_config_path": str(self.base_config_path),
            "base_checkpoint_path": self.base_checkpoint_relative,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_config_sha256": self.base_config_sha256,
            "checkpoint_selection": "exact_milestone",
            "frozen": True,
        }


class CaarSwitcherCandidateConfig(AlgoBase, extra=Extra.forbid):
    """Explicit deployment fields for one selected CAAR milestone."""

    name: Literal["CAAR-Candidate"] = "CAAR-Candidate"
    path_to_weights: str
    milestone_checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    base_weights_path: str
    base_checkpoint_path: str
    base_checkpoint_sha256: str
    base_config_sha256: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": CAAR_CANDIDATE_KIND,
            "schema": CAAR_CANDIDATE_SCHEMA,
            "weights_path": self.path_to_weights,
            "checkpoint_path": self.milestone_checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "base_weights_path": self.base_weights_path,
            "base_checkpoint_path": self.base_checkpoint_path,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_config_sha256": self.base_config_sha256,
            "checkpoint_selection": "exact_milestone",
            "frozen": True,
        }


class CaarSwitcherCandidate:
    """Runtime adapter for a hash-pinned, inference-only CAAR policy."""

    def __init__(
        self,
        policy: EPOMTraceContext,
        artifact: CaarCandidateArtifact,
        *,
        verified_file_hashes: Mapping[str, str] | None = None,
    ):
        self.policy = policy
        self.artifact = artifact
        self._verified_file_hashes = dict(
            verified_file_hashes
            if verified_file_hashes is not None
            else artifact.verify_files()
        )
        self.ppo.eval()
        for parameter in self.ppo.parameters():
            parameter.requires_grad_(False)
        self._verify_loaded_provenance()

    @classmethod
    def load(
        cls,
        artifact: CaarCandidateArtifact,
        *,
        seed: int,
        device: str,
    ) -> "CaarSwitcherCandidate":
        verified = artifact.verify_files()
        policy = EPOMTraceContext(
            EPOMTraceContextConfig(
                path_to_weights=str(artifact.weights_path),
                checkpoint_kind="milestone",
                milestone_checkpoint=str(artifact.checkpoint_path),
                seed=int(seed),
                device=str(device),
            )
        )
        return cls(policy, artifact, verified_file_hashes=verified)

    @property
    def ppo(self):
        return self.policy.ppo

    @property
    def device(self):
        return self.policy.device

    def _verify_loaded_provenance(self) -> None:
        provenance = self.policy.get_model_provenance()
        model = provenance.get("model", {})
        actual = {
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "config_sha256": provenance.get("config_sha256"),
            "base_checkpoint_sha256": model.get("base_checkpoint_sha256"),
            "base_config_sha256": model.get("base_config_sha256"),
        }
        expected = {
            "checkpoint_sha256": self.artifact.checkpoint_sha256,
            "config_sha256": self.artifact.config_sha256,
            "base_checkpoint_sha256": self.artifact.base_checkpoint_sha256,
            "base_config_sha256": self.artifact.base_config_sha256,
        }
        if actual != expected:
            raise RuntimeError(
                f"Loaded CAAR identity differs: expected={expected}, actual={actual}"
            )
        if model.get("trace_architecture") != CAAR_TRACE_ARCHITECTURE:
            raise RuntimeError(
                "Loaded candidate is not the selected paper CAAR architecture: "
                f"{model.get('trace_architecture')!r}"
            )
        if model.get("actor_backbone_tensor_sha256_verified") is not True:
            raise RuntimeError("Frozen EPOM-L actor verification is not true.")

    def verify_frozen(self, *, rehash_files: bool = False) -> dict[str, object]:
        if rehash_files:
            current = self.artifact.verify_files()
            if current != self._verified_file_hashes:
                raise RuntimeError("Pinned CAAR files changed after load.")
        trainable = [
            name
            for name, parameter in self.ppo.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"Frozen CAAR exposes trainable parameters: {trainable}")
        if self.ppo.training:
            raise RuntimeError("Frozen CAAR was switched to training mode.")
        self._verify_loaded_provenance()
        return {
            "verified": True,
            "trainable_parameter_count": 0,
            "file_sha256": deepcopy(self._verified_file_hashes),
        }

    def set_grid_config(self, grid_config) -> None:
        self.policy.set_grid_config(grid_config)

    def set_env(self, env) -> None:
        self.policy.set_env(env)

    def after_reset(self) -> None:
        self.policy.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        return self.policy.act(observations, rewards, dones, infos)

    def after_step(self, dones) -> None:
        self.policy.after_step(dones)

    def get_action_correction_stats(self) -> dict:
        provider = getattr(self.policy, "get_action_correction_stats", None)
        return provider() if callable(provider) else {}

    def get_model_provenance(self) -> dict[str, object]:
        return {
            "schema": CAAR_CANDIDATE_SCHEMA,
            "candidate": deepcopy(self.artifact.as_dict()),
            "frozen_verification": self.verify_frozen(),
            "underlying": deepcopy(self.policy.get_model_provenance()),
        }


__all__ = [
    "CAAR_CANDIDATE_KIND",
    "CAAR_CANDIDATE_LABEL",
    "CAAR_CANDIDATE_SCHEMA",
    "CAAR_TRACE_ARCHITECTURE",
    "CaarCandidateArtifact",
    "CaarSwitcherCandidate",
    "CaarSwitcherCandidateConfig",
]
