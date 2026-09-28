"""Explicit deployment correction; never part of checkpoint training state."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class InferenceCorrection:
    """Use only the learned residual when raw base entropy exceeds a threshold."""

    entropy_threshold: float = 0.01
    residual_scale: float = 12.0
    action_sampling: str = "direct_numpy"

    def __post_init__(self):
        if not math.isfinite(self.entropy_threshold) or self.entropy_threshold < 0:
            raise ValueError("Inference entropy_threshold must be finite and nonnegative.")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("Inference residual_scale must be finite and nonnegative.")
        if self.action_sampling not in {"torch", "direct_numpy"}:
            raise ValueError("Unsupported inference action_sampling.")

    def gate(self, entropy):
        return (entropy > self.entropy_threshold).unsqueeze(-1)

    def apply(self, base_logits, residual, entropy):
        scale = self.gate(entropy).to(base_logits) * self.residual_scale
        learned_delta = scale * residual
        return base_logits + learned_delta, learned_delta, torch.zeros_like(scale), entropy

    def as_dict(self):
        return asdict(self)

    def provenance(self):
        return dict(self.as_dict(), direct_bonus=0.0, comparison=">",
                    formula="base_logits + scale * center5(0.5*tanh(raw))",
                    inactive_policy="pure_base", entropy_normalization="none",
                    checkpoint_training_settings_unchanged=True)
