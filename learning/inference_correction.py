from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class InferenceCorrection:
    entropy_threshold: float = 0.01
    action_sampling: str = "direct_numpy"

    def __post_init__(self):
        if not math.isfinite(self.entropy_threshold) or self.entropy_threshold < 0:
            raise ValueError(
                "Inference entropy_threshold must be finite and nonnegative."
            )
        if self.action_sampling not in {"torch", "direct_numpy"}:
            raise ValueError("Unsupported inference action_sampling.")

    def gate(self, entropy):
        return (entropy > self.entropy_threshold).unsqueeze(-1)

    def apply(self, base_logits, residual, entropy):
        gate = self.gate(entropy).to(base_logits)
        learned_delta = gate * residual
        return (
            base_logits + learned_delta,
            learned_delta,
            torch.zeros_like(gate),
            entropy,
        )

    def as_dict(self):
        return asdict(self)
