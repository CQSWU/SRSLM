"""Trace variants for the B1/B2/B3 causal-attribution controls.

All three variants keep the observation, the network and the number of
trainable parameters byte-identical.  Only the contents of the ``tau`` channel
differ, and the variant is fixed for the whole of stage-two training so the
control groups are trained, not merely evaluated:

``real``      the crop centred on the agent's own position (the method).
``zero``      an all-zero crop.  Isolates whatever the residual branch can do
              from its extra capacity and from the ``agents`` / ``h`` inputs
              alone.
``shuffled``  a crop of the *same* global trace at a decorrelated location --
              by default another agent's current position, resampled every
              step.  Map, agent density, numeric distribution and local spatial
              structure all match the real crop; only the correspondence with
              this agent's own state is broken.  A random pixel permutation
              would destroy the spatial structure and the input distribution at
              the same time and could not separate the two effects.
"""

import numpy as np


VARIANTS = ("real", "zero", "shuffled")


class TraceVariant:
    """Rewrite the ``tau`` field of a batch of observations in place."""

    def __init__(self, variant="real", seed=None, shuffle_mode="other_agent"):
        variant = str(variant)
        if variant not in VARIANTS:
            raise ValueError(f"trace variant must be one of {VARIANTS}.")
        if shuffle_mode not in ("other_agent", "random_free"):
            raise ValueError(
                "shuffle_mode must be 'other_agent' or 'random_free'."
            )
        self.variant = variant
        self.shuffle_mode = shuffle_mode
        self.rng = np.random.default_rng(seed)

    def wants_alternate_positions(self):
        return self.variant == "shuffled"

    def alternate_positions(self, positions, free_cells=None):
        """Decorrelated read positions, one per agent."""
        positions = np.asarray(positions, dtype=np.int64)
        count = len(positions)
        if self.shuffle_mode == "random_free" and free_cells is not None:
            picks = self.rng.integers(0, len(free_cells), size=count)
            return np.asarray(free_cells, dtype=np.int64)[picks]

        if count < 2:
            return positions
        # Another agent's position: a derangement, so no agent reads its own.
        offset = self.rng.integers(1, count, size=count)
        return positions[(np.arange(count) + offset) % count]

    def apply(self, observations):
        if self.variant == "zero":
            for observation in observations:
                tau = observation.get("tau")
                if tau is not None:
                    observation["tau"] = np.zeros_like(tau)
        return observations

    def describe(self):
        return {
            "trace_variant": self.variant,
            "shuffle_mode": (
                self.shuffle_mode if self.variant == "shuffled" else None
            ),
        }


__all__ = ["TraceVariant", "VARIANTS"]
