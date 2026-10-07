from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from pydantic import Extra

from agents.epom_trace_context import EPOMTraceContext, EPOMTraceContextConfig
from agents.utils_agents import AlgoBase
from learning.inference_correction import InferenceCorrection


@dataclass(frozen=True)
class ArpeCandidateArtifact:
    project_root: Path
    weights_path: Path
    checkpoint_path: Path
    base_weights_path: Path
    base_checkpoint_path: Path

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        project_root: Path,
    ) -> "ArpeCandidateArtifact":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            **{
                field: (root / str(mapping[field])).resolve()
                for field in (
                    "weights_path",
                    "checkpoint_path",
                    "base_weights_path",
                    "base_checkpoint_path",
                )
            },
        )

    @classmethod
    def from_config(
        cls,
        config: "ARPEConfig",
        project_root: Path,
    ) -> "ArpeCandidateArtifact":
        return cls.from_mapping(config.as_mapping(), project_root)

    def as_dict(self) -> dict[str, object]:
        result = {}
        for field in (
            "weights_path",
            "checkpoint_path",
            "base_weights_path",
            "base_checkpoint_path",
        ):
            path = getattr(self, field)
            result[field] = (
                path.relative_to(self.project_root).as_posix()
                if path.is_relative_to(self.project_root)
                else str(path)
            )
        return result


class ARPEConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["ARPE"] = "ARPE"
    path_to_weights: str
    milestone_checkpoint: str
    base_weights_path: str
    base_checkpoint_path: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "weights_path": self.path_to_weights,
            "checkpoint_path": self.milestone_checkpoint,
            "base_weights_path": self.base_weights_path,
            "base_checkpoint_path": self.base_checkpoint_path,
        }


class ARPE:
    def __init__(
        self,
        policy: EPOMTraceContext,
        artifact: ArpeCandidateArtifact,
    ):
        self.policy = policy
        self.artifact = artifact
        self.ppo.eval()
        for parameter in self.ppo.parameters():
            parameter.requires_grad_(False)
        self.ppo.inference_correction = InferenceCorrection()

    @classmethod
    def load(
        cls,
        artifact: ArpeCandidateArtifact,
        *,
        seed: int,
        device: str,
    ) -> "ARPE":
        policy = EPOMTraceContext(
            EPOMTraceContextConfig(
                path_to_weights=str(artifact.weights_path),
                milestone_checkpoint=str(artifact.checkpoint_path),
                seed=int(seed),
                device=str(device),
                base_weights_path=str(artifact.base_weights_path),
                base_checkpoint_path=str(artifact.base_checkpoint_path),
            )
        )
        return cls(policy, artifact)

    @property
    def ppo(self):
        return self.policy.ppo

    @property
    def device(self):
        return self.policy.device

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


__all__ = [
    "ArpeCandidateArtifact",
    "ARPE",
    "ARPEConfig",
]
