"""Checkpoint-backed inference policy for the SRSLM Switcher."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Literal, Mapping

import gymnasium as gym
import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from agents.switcher_core import (
    NUM_BRANCHES,
    SWITCHER_FIELD_SHAPES,
    switcher_observation_space,
)
from agents.arpe import ArpeCandidateArtifact
from agents.utils_agents import AlgoBase
from train import register_custom_components, validate_config


class SwitcherConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["Switcher"] = "Switcher"
    path_to_weights: str = "weights/SRSLM-Switcher-Final-1B"
    checkpoint_kind: Literal["auto", "latest", "best"] = "auto"
    deterministic: bool = False


class Switcher:
    """Choose between ARPE and AORePlan for non-wait AORePlan actions."""

    expected_encoder_custom = "switcher"
    policy_label = "Switcher"

    def __init__(self, cfg: SwitcherConfig):
        self.cfg = cfg
        path = Path(cfg.path_to_weights)
        self.config_path = (path / "config.json").resolve()
        register_custom_components()
        payload = self.config_path.read_bytes()
        config = json.loads(payload.decode("utf-8"))
        full_config = deepcopy(config["full_config"])
        declaration = full_config.pop("candidate_policy", None)
        if declaration is not None and not isinstance(declaration, dict):
            raise RuntimeError("Switcher candidate_policy declaration is malformed.")
        candidate_artifact = None
        if declaration is not None:
            project_root = Path(__file__).resolve().parents[1]
            candidate_artifact = ArpeCandidateArtifact.from_mapping(
                declaration,
                project_root,
            )
        from learning.config import checkpoint_experiment_config

        _, flat_config = validate_config(checkpoint_experiment_config(full_config))
        if flat_config.encoder_custom != self.expected_encoder_custom:
            raise RuntimeError(
                f"Checkpoint is not a {self.policy_label} policy: expected "
                f"encoder_custom={self.expected_encoder_custom!r}."
            )
        if bool(flat_config.use_rnn):
            raise RuntimeError("Switcher checkpoint must be feed-forward.")

        observation_space = gym.spaces.Dict(
            {
                **switcher_observation_space().spaces,
                "switch_allowed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        action_space = gym.spaces.Discrete(NUM_BRANCHES)
        actor = create_actor_critic(flat_config, observation_space, action_space)
        self.device = self._resolve_device(cfg.device)
        actor.model_to_device(self.device)

        checkpoint_dir = join(str(path), f"checkpoint_p{flat_config.policy_index}")
        checkpoint_path = self._resolve_checkpoint(
            checkpoint_dir,
            cfg.checkpoint_kind,
        )
        checkpoint_payload = checkpoint_path.read_bytes()
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(
            io.BytesIO(checkpoint_payload),
            map_location=self.device,
            weights_only=False,
        )
        actor.load_state_dict(checkpoint["model"])
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

        self.ppo = actor
        self.flat_config = flat_config
        self.candidate_artifact = candidate_artifact
        self.after_reset()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        requested = str(requested).lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_checkpoint(checkpoint_dir, kind: str) -> Path:
        directory = Path(checkpoint_dir)
        if kind == "best":
            candidates = sorted(directory.glob("best_*avg_throughput*.pth"))
            if not candidates:
                candidates = sorted(directory.glob("best_*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No best Switcher checkpoint in {directory}.")
            path = candidates[-1]
            label = "best"
        else:
            candidates = Learner.get_checkpoints(str(directory))
            if candidates:
                path = Path(candidates[-1])
                label = "latest"
            elif kind == "auto":
                best = sorted(directory.glob("best_*.pth"))
                if not best:
                    raise FileNotFoundError(f"No Switcher checkpoint in {directory}.")
                path = best[-1]
                label = "best"
            else:
                raise FileNotFoundError(
                    f"No latest Switcher checkpoint in {directory}."
                )
        path = path.resolve()
        log.info("Loading %s Switcher checkpoint: %s", label, path)
        return path

    def after_reset(self) -> None:
        torch.manual_seed(int(self.cfg.seed or 0))
        self.total_choice_count = 0
        self.ao_choice_count = 0
        self._ao_probability_samples = []

    def choose(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        expected = SWITCHER_FIELD_SHAPES
        arrays = {}
        count = None
        for key, trailing_shape in expected.items():
            if key not in state:
                raise ValueError(f"Switcher state is missing {key!r}.")
            array = np.asarray(state[key], dtype=np.float32)
            if (
                array.ndim != len(trailing_shape) + 1
                or tuple(array.shape[1:]) != trailing_shape
            ):
                raise ValueError(
                    f"Switcher field {key!r} expected [N,{trailing_shape}], "
                    f"got {array.shape}."
                )
            if count is None:
                count = len(array)
            elif len(array) != count:
                raise ValueError("Switcher state fields have different batches.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Switcher field {key!r} is non-finite.")
            arrays[key] = array
        arrays["switch_allowed"] = np.asarray(
            state.get("switch_allowed", arrays["aoreplan_action"][:, :1] < 0.5),
            dtype=np.float32,
        )
        if arrays["switch_allowed"].shape != (count, 1) or not np.all(
            np.isin(arrays["switch_allowed"], (0.0, 1.0))
        ):
            raise ValueError("Switcher switch_allowed must be a binary [N,1] mask.")
        if not count:
            raise ValueError("Switcher received an empty batch.")
        if not np.all(arrays["aoreplan_action"][:, 0] == 0.0):
            raise ValueError("Only non-wait AORePlan states may enter Switcher.")
        rnn_states = torch.zeros(
            (count, get_rnn_size(self.flat_config)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            observations = TensorDict(
                {
                    key: torch.as_tensor(
                        value,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for key, value in arrays.items()
                }
            )
            observations = prepare_and_normalize_obs(self.ppo, observations)
            outputs = self.ppo(observations, rnn_states)
            logits = outputs["action_logits"]
            probabilities = torch.softmax(logits, dim=-1)
            if self.cfg.deterministic:
                actions = torch.argmax(probabilities, dim=-1)
            else:
                actions = outputs["actions"]
            result = actions.detach().cpu().numpy().astype(np.int64)
            ao_probabilities = probabilities[:, 1].detach().cpu().numpy()

        self.total_choice_count += count
        self.ao_choice_count += int(np.sum(result == 1))
        self._ao_probability_samples.append(ao_probabilities)
        return result

    def get_stats(self) -> dict:
        if self._ao_probability_samples:
            probabilities = np.concatenate(self._ao_probability_samples)
            mean_probability = float(probabilities.mean())
            p05 = float(np.quantile(probabilities, 0.05))
            p95 = float(np.quantile(probabilities, 0.95))
        else:
            mean_probability = p05 = p95 = 0.0
        result = {
            "switcher_stochastic": not self.cfg.deterministic,
            "switcher_model_choice_count": self.total_choice_count,
            "switcher_model_selected_ao_count": self.ao_choice_count,
            "switcher_sampled_ao_rate": (
                self.ao_choice_count / self.total_choice_count
                if self.total_choice_count
                else 0.0
            ),
            "switcher_ao_probability_mean": mean_probability,
            "switcher_ao_probability_p05": p05,
            "switcher_ao_probability_p95": p95,
        }
        return result


__all__ = [
    "Switcher",
    "SwitcherConfig",
]
