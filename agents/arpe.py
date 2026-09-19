"""Frozen ARPE adapter shared by Switcher training and evaluation.

Only the weight directories and checkpoint files are required.  Hashes are
recorded as provenance after loading, but users are not required to reproduce
the paper artifact hashes before running or retraining the code.
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


# Serialized identifiers stay exact for the selected historical checkpoints.
# Current method names and entrypoints are ARPE; see docs/METHOD_NAMING.md.
ARPE_CANDIDATE_KIND = "epom_trace_context_caar_milestone"
ARPE_CANDIDATE_LABEL = "ARPE"
ARPE_CANDIDATE_SCHEMA = "switcher_candidate_caar_v1"
ARPE_TRACE_ARCHITECTURE = (
    "trace_allaction_centered_tanh_residual200_failcredit_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(project_root: Path, value: object, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ARPE candidate {field} must be a non-empty path string.")
    declared = Path(value)
    root = Path(project_root).resolve()
    resolved = declared.resolve() if declared.is_absolute() else (root / declared).resolve()
    return str(declared), resolved


@dataclass(frozen=True)
class ArpeCandidateArtifact:
    """Immutable identity of the ARPE policy used as Switcher branch zero."""

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

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        project_root: Path,
    ) -> "ArpeCandidateArtifact":
        required = {
            "weights_path",
            "checkpoint_path",
            "base_weights_path",
            "base_checkpoint_path",
        }
        missing = required - set(mapping)
        if missing:
            raise ValueError(
                "ARPE candidate declaration is missing: "
                f"{sorted(missing)}"
            )
        for field, expected in (
            ("kind", ARPE_CANDIDATE_KIND),
            ("schema", ARPE_CANDIDATE_SCHEMA),
            ("frozen", True),
        ):
            if field in mapping and mapping[field] != expected:
                raise ValueError(
                    f"ARPE candidate {field} must be {expected!r}, got {mapping[field]!r}."
                )

        root = Path(project_root).resolve()
        weights_relative, weights_path = _artifact_path(
            root, mapping["weights_path"], "weights_path"
        )
        checkpoint_relative, checkpoint_path = _artifact_path(
            root, mapping["checkpoint_path"], "checkpoint_path"
        )
        base_weights_relative, base_weights_path = _artifact_path(
            root, mapping["base_weights_path"], "base_weights_path"
        )
        base_checkpoint_relative, base_checkpoint_path = _artifact_path(
            root, mapping["base_checkpoint_path"], "base_checkpoint_path"
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
        )

    @classmethod
    def from_config(
        cls,
        config: "ARPEConfig",
        project_root: Path,
    ) -> "ArpeCandidateArtifact":
        return cls.from_mapping(config.as_mapping(), project_root)

    def inspect_files(self) -> dict[str, str]:
        """Check that inputs exist and return hashes for optional provenance."""
        files = (
            self.config_path,
            self.checkpoint_path,
            self.base_config_path,
            self.base_checkpoint_path,
        )
        inspected: dict[str, str] = {}
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"ARPE input is missing: {path}")
            inspected[str(path)] = _sha256(path)
        return inspected

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": ARPE_CANDIDATE_KIND,
            "label": ARPE_CANDIDATE_LABEL,
            "schema": ARPE_CANDIDATE_SCHEMA,
            "weights_path": self.weights_relative,
            "config_path": str(self.config_path),
            "checkpoint_path": self.checkpoint_relative,
            "base_weights_path": self.base_weights_relative,
            "base_config_path": str(self.base_config_path),
            "base_checkpoint_path": self.base_checkpoint_relative,
            "frozen": True,
        }


class ARPEConfig(AlgoBase, extra=Extra.forbid):
    """Explicit deployment fields for one selected ARPE milestone."""

    name: Literal["ARPE"] = "ARPE"
    path_to_weights: str
    milestone_checkpoint: str
    base_weights_path: str
    base_checkpoint_path: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": ARPE_CANDIDATE_KIND,
            "schema": ARPE_CANDIDATE_SCHEMA,
            "weights_path": self.path_to_weights,
            "checkpoint_path": self.milestone_checkpoint,
            "base_weights_path": self.base_weights_path,
            "base_checkpoint_path": self.base_checkpoint_path,
            "frozen": True,
        }


class ARPE:
    """Runtime adapter for an inference-only ARPE policy."""

    def __init__(
        self,
        policy: EPOMTraceContext,
        artifact: ArpeCandidateArtifact,
        *,
        verified_file_hashes: Mapping[str, str] | None = None,
    ):
        self.policy = policy
        self.artifact = artifact
        self._verified_file_hashes = dict(
            verified_file_hashes
            if verified_file_hashes is not None
            else artifact.inspect_files()
        )
        self._verify_loaded_artifacts()
        self.ppo.eval()
        for parameter in self.ppo.parameters():
            parameter.requires_grad_(False)

    def _verify_loaded_artifacts(self, file_hashes: Mapping[str, str] | None = None) -> None:
        """The declaration must describe what was loaded, not merely existing files."""
        declared = self._verified_file_hashes if file_hashes is None else file_hashes
        model = self.ppo.checkpoint_provenance()
        actual = {
            str(self.artifact.config_path): self.policy.config_sha256,
            str(self.artifact.checkpoint_path): self.policy.checkpoint_sha256,
            str(self.artifact.base_config_path): model["base_config_sha256"],
            str(self.artifact.base_checkpoint_path): model["base_checkpoint_sha256"],
        }
        mismatches = {
            path: {"declared": declared.get(path), "loaded": digest}
            for path, digest in actual.items()
            if declared.get(path) != digest
        }
        if mismatches:
            raise RuntimeError(
                "ARPE loaded artifacts differ from its declaration; refusing to "
                f"report misleading provenance: {mismatches}"
            )

    @classmethod
    def load(
        cls,
        artifact: ArpeCandidateArtifact,
        *,
        seed: int,
        device: str,
        action_sampling: Literal["torch", "direct_numpy"] = "torch",
    ) -> "ARPE":
        verified = artifact.inspect_files()
        policy = EPOMTraceContext(
            EPOMTraceContextConfig(
                path_to_weights=str(artifact.weights_path),
                checkpoint_kind="milestone",
                milestone_checkpoint=str(artifact.checkpoint_path),
                seed=int(seed),
                device=str(device),
                action_sampling=action_sampling,
            )
        )
        return cls(policy, artifact, verified_file_hashes=verified)

    @property
    def ppo(self):
        return self.policy.ppo

    @property
    def device(self):
        return self.policy.device

    def verify_frozen(self, *, rehash_files: bool = False) -> dict[str, object]:
        if rehash_files:
            hashes = self.artifact.inspect_files()
            self._verify_loaded_artifacts(hashes)
            self._verified_file_hashes = hashes
        trainable = [
            name
            for name, parameter in self.ppo.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"Frozen ARPE exposes trainable parameters: {trainable}")
        if self.ppo.training:
            raise RuntimeError("Frozen ARPE was switched to training mode.")
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
            "schema": ARPE_CANDIDATE_SCHEMA,
            "candidate": deepcopy(self.artifact.as_dict()),
            "frozen_verification": self.verify_frozen(),
            "underlying": deepcopy(self.policy.get_model_provenance()),
        }


__all__ = [
    "ARPE_CANDIDATE_KIND",
    "ARPE_CANDIDATE_LABEL",
    "ARPE_CANDIDATE_SCHEMA",
    "ARPE_TRACE_ARCHITECTURE",
    "ArpeCandidateArtifact",
    "ARPE",
    "ARPEConfig",
]
