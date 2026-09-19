"""The real shared trace and its parameter-matched zero-input control."""

import numpy as np


VARIANTS = ("real", "zero")


class TraceVariant:
    """Rewrite the ``tau`` field of a batch of observations in place."""

    def __init__(self, variant="real"):
        variant = str(variant)
        if variant not in VARIANTS:
            raise ValueError(f"trace variant must be one of {VARIANTS}.")
        self.variant = variant

    def apply(self, observations):
        if self.variant == "zero":
            for observation in observations:
                observation["tau"] = np.zeros_like(observation["tau"])
        return observations

__all__ = ["TraceVariant", "VARIANTS"]
