"""Independent trace-aware MC-return estimators for CAAR and AO-safe.

This module deliberately implements policy evaluation, not a switching actor.
Each checkpoint contains one scalar state-value regressor trained from the raw
Monte-Carlo returns of exactly one fixed behavior policy.  The estimator has
no action output and no recurrent state.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from learning.epom_encoder import (
    EPOMEncoder,
    SUPPORTED_COORDINATE_ENCODINGS,
)


CHECKPOINT_SCHEMA_VERSION = "caar_policy_return_estimator_shared_trace_v1"
ESTIMATOR_KIND = "absolute_raw_mc_return_with_shared_trace"
SUPPORTED_BRANCHES = ("caar", "ao_safe")
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "estimator_kind",
        "branch",
        "model_config",
        "model_state_dict",
        "training_metadata",
    }
)


class CheckpointSchemaError(ValueError):
    """Raised when a policy-return checkpoint violates the public schema."""


@dataclass(frozen=True)
class PolicyEstimationModelConfig:
    """Architecture recorded verbatim in every estimator checkpoint."""

    obs_shape: tuple[int, int, int] = (4, 11, 11)
    hidden_size: int = 512
    encoder_num_filters: int = 64
    encoder_num_res_blocks: int = 3
    encoder_extra_fc_layers: int = 1
    nonlinearity: str = "relu"
    coordinate_encoding: str = "absolute_v1"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.obs_shape)
        if len(shape) != 3 or shape[0] != 4:
            raise ValueError(
                "obs_shape must be (4, height, width), with Shared Traffic "
                "Trace as the fourth channel."
            )
        if shape[1] < 1 or shape[2] < 1:
            raise ValueError("Observation height and width must be positive.")
        if int(self.hidden_size) != 512:
            raise ValueError(
                "The original-style policy estimator requires hidden_size=512."
            )
        if int(self.encoder_num_filters) < 1:
            raise ValueError("encoder_num_filters must be positive.")
        if int(self.encoder_num_res_blocks) < 0:
            raise ValueError("encoder_num_res_blocks must be non-negative.")
        if int(self.encoder_extra_fc_layers) < 1:
            raise ValueError("encoder_extra_fc_layers must be positive.")
        if self.nonlinearity != "relu":
            raise ValueError(
                "The faithful EPOM-style estimator uses ReLU nonlinearity."
            )
        if self.coordinate_encoding not in SUPPORTED_COORDINATE_ENCODINGS:
            raise ValueError(
                "coordinate_encoding must be one of "
                f"{SUPPORTED_COORDINATE_ENCODINGS}, got "
                f"{self.coordinate_encoding!r}."
            )
        object.__setattr__(self, "obs_shape", shape)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["obs_shape"] = list(self.obs_shape)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyEstimationModelConfig":
        if not isinstance(value, Mapping):
            raise CheckpointSchemaError("model_config must be a mapping.")
        expected = frozenset(cls.__dataclass_fields__)
        # Preserve the frozen runtime's legacy-key compatibility.  New trace
        # checkpoints always write coordinate_encoding explicitly.
        legacy = expected - {"coordinate_encoding"}
        actual = frozenset(value)
        if actual != expected and actual != legacy:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise CheckpointSchemaError(
                "model_config keys do not match the schema; "
                f"missing={missing}, unexpected={unexpected}."
            )
        try:
            return cls(
                obs_shape=tuple(value["obs_shape"]),
                hidden_size=int(value["hidden_size"]),
                encoder_num_filters=int(value["encoder_num_filters"]),
                encoder_num_res_blocks=int(value["encoder_num_res_blocks"]),
                encoder_extra_fc_layers=int(
                    value["encoder_extra_fc_layers"]
                ),
                nonlinearity=str(value["nonlinearity"]),
                coordinate_encoding=str(
                    value.get("coordinate_encoding", "absolute_v1")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointSchemaError(
                f"Invalid model_config: {exc}"
            ) from exc


def _epom_cfg(config: PolicyEstimationModelConfig) -> SimpleNamespace:
    """Create the minimal Sample Factory config consumed by EPOMEncoder."""

    return SimpleNamespace(
        hidden_size=config.hidden_size,
        encoder_extra_fc_layers=config.encoder_extra_fc_layers,
        nonlinearity=config.nonlinearity,
        coordinate_encoding=config.coordinate_encoding,
        use_spectral_norm=False,
        full_config={
            "experiment_settings": {
                "pogema_encoder_num_filters": config.encoder_num_filters,
                "pogema_encoder_num_res_blocks": (
                    config.encoder_num_res_blocks
                ),
            }
        },
    )


def _observation_space(config: PolicyEstimationModelConfig) -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(
                # The first three channels are binary; the fourth is signed
                # mean-centred pressure and is intentionally unbounded here.
                low=-np.inf,
                high=np.inf,
                shape=config.obs_shape,
                dtype=np.float32,
            ),
            "xy": gym.spaces.Box(
                low=-1024.0,
                high=1024.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "target_xy": gym.spaces.Box(
                low=-1024.0,
                high=1024.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )


class PolicyEstimationModel(nn.Module):
    """EPOM-style non-recurrent encoder followed by a scalar value head."""

    def __init__(
        self,
        config: PolicyEstimationModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = config or PolicyEstimationModelConfig()
        cfg = _epom_cfg(self.model_config)
        self.encoder = EPOMEncoder(cfg, _observation_space(self.model_config))
        encoder_size = int(self.encoder.get_encoder_out_size())
        # This is the original LSwitcher head: two 512-unit MLP layers and
        # one scalar absolute-return output.
        self.value_head = nn.Sequential(
            nn.Linear(encoder_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    @staticmethod
    def _validate_observations(observations: Mapping[str, torch.Tensor]) -> None:
        expected_keys = {"obs", "xy", "target_xy"}
        actual_keys = set(observations)
        if actual_keys != expected_keys:
            raise ValueError(
                "Estimator observations require exactly obs, xy, target_xy; "
                f"got {sorted(actual_keys)}."
            )
        obs = observations["obs"]
        xy = observations["xy"]
        target_xy = observations["target_xy"]
        if obs.ndim != 4:
            raise ValueError("obs must have shape (batch, 4, height, width).")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (batch, 2).")
        if target_xy.ndim != 2 or target_xy.shape[1] != 2:
            raise ValueError("target_xy must have shape (batch, 2).")
        if not (obs.shape[0] == xy.shape[0] == target_xy.shape[0]):
            raise ValueError("Estimator observation batch sizes do not match.")

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate_observations(observations)
        expected = self.model_config.obs_shape
        if tuple(observations["obs"].shape[1:]) != expected:
            raise ValueError(
                "obs shape does not match this checkpoint: expected "
                f"{expected}, got {tuple(observations['obs'].shape[1:])}."
            )
        encoded = self.encoder(observations)
        return self.value_head(encoded).squeeze(-1)


def _validate_branch(branch: str) -> str:
    branch = str(branch)
    if branch not in SUPPORTED_BRANCHES:
        raise CheckpointSchemaError(
            f"branch must be one of {SUPPORTED_BRANCHES}, got {branch!r}."
        )
    return branch


def make_checkpoint(
    model: PolicyEstimationModel,
    *,
    branch: str,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only accepted checkpoint payload for this estimator."""

    branch = _validate_branch(branch)
    metadata = dict(training_metadata or {})
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "estimator_kind": ESTIMATOR_KIND,
        "branch": branch,
        "model_config": model.model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "training_metadata": metadata,
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_branch: str | None = None,
) -> tuple[str, PolicyEstimationModelConfig]:
    """Validate every schema field before a model is constructed."""

    if not isinstance(payload, Mapping):
        raise CheckpointSchemaError("Checkpoint payload must be a mapping.")
    actual_keys = frozenset(payload)
    if actual_keys != _CHECKPOINT_KEYS:
        raise CheckpointSchemaError(
            "Checkpoint keys do not match the schema; "
            f"missing={sorted(_CHECKPOINT_KEYS - actual_keys)}, "
            f"unexpected={sorted(actual_keys - _CHECKPOINT_KEYS)}."
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointSchemaError(
            "Unsupported checkpoint schema: "
            f"{payload['schema_version']!r}."
        )
    if payload["estimator_kind"] != ESTIMATOR_KIND:
        raise CheckpointSchemaError(
            "Checkpoint is not a trace-aware absolute raw-MC return estimator."
        )
    branch = _validate_branch(payload["branch"])
    if expected_branch is not None and branch != _validate_branch(expected_branch):
        raise CheckpointSchemaError(
            f"Expected {expected_branch!r} checkpoint, found {branch!r}."
        )
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointSchemaError("model_state_dict must be a mapping.")
    if not isinstance(payload["training_metadata"], Mapping):
        raise CheckpointSchemaError("training_metadata must be a mapping.")
    config = PolicyEstimationModelConfig.from_dict(payload["model_config"])
    metadata_encoding = payload["training_metadata"].get("coordinate_encoding")
    if (
        metadata_encoding is not None
        and str(metadata_encoding) != config.coordinate_encoding
    ):
        raise CheckpointSchemaError(
            "training_metadata coordinate_encoding disagrees with model_config."
        )
    return branch, config


def save_policy_return_checkpoint(
    path: str | Path,
    model: PolicyEstimationModel,
    *,
    branch: str,
    training_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a strict estimator checkpoint."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_checkpoint(
        model,
        branch=branch,
        training_metadata=training_metadata,
    )
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "gpu":
        requested = "cuda"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


@contextmanager
def _preserve_torch_rng():
    """Prevent model construction or prediction from perturbing caller RNG."""

    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _stack_matrix_observations(
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if isinstance(observations, Mapping):
        values = observations
        result = {
            key: torch.as_tensor(values[key], device=device, dtype=torch.float32)
            for key in ("obs", "xy", "target_xy")
        }
    else:
        sequence = tuple(observations)
        if not sequence:
            raise ValueError("predict() requires at least one observation.")
        result = {
            key: torch.as_tensor(
                np.stack([np.asarray(value[key]) for value in sequence]),
                device=device,
                dtype=torch.float32,
            )
            for key in ("obs", "xy", "target_xy")
        }
    if result["obs"].ndim == 3:
        result["obs"] = result["obs"].unsqueeze(0)
    if result["xy"].ndim == 1:
        result["xy"] = result["xy"].unsqueeze(0)
    if result["target_xy"].ndim == 1:
        result["target_xy"] = result["target_xy"].unsqueeze(0)
    return result


class PolicyReturnEstimator(nn.Module):
    """Strict, deterministic inference wrapper for one policy estimator."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "auto",
        expected_branch: str | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = resolve_device(device)
        with _preserve_torch_rng():
            payload = _torch_load(self.checkpoint_path)
            self.branch, config = validate_checkpoint_payload(
                payload,
                expected_branch=expected_branch,
            )
            self.model = PolicyEstimationModel(config)
            try:
                self.model.load_state_dict(
                    payload["model_state_dict"],
                    strict=True,
                )
            except RuntimeError as exc:
                raise CheckpointSchemaError(
                    "Checkpoint weights do not match model_config."
                ) from exc
            self.model.to(self.device)
            self.model.eval()
        self.training_metadata = dict(payload["training_metadata"])
        self.requires_grad_(False)
        self.eval()

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.model(observations)

    def predict(
        self,
        matrix_observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> np.ndarray:
        """Predict absolute returns without sampling or changing global RNG."""

        with _preserve_torch_rng(), torch.inference_mode():
            batch = _stack_matrix_observations(
                matrix_observations,
                device=self.device,
            )
            values = self.model(batch)
            return values.detach().cpu().numpy().astype(np.float32, copy=True)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "ESTIMATOR_KIND",
    "SUPPORTED_COORDINATE_ENCODINGS",
    "SUPPORTED_BRANCHES",
    "CheckpointSchemaError",
    "PolicyEstimationModel",
    "PolicyEstimationModelConfig",
    "PolicyReturnEstimator",
    "make_checkpoint",
    "resolve_device",
    "save_policy_return_checkpoint",
    "validate_checkpoint_payload",
]
