"""Offline policy-return estimation for the CAAR/AO-safe switcher.

The package intentionally contains supervised data collection utilities only.
It does not train a switching policy and does not depend on the retired PPO
gate implementation.
"""

from policy_estimation.dataset import (
    DATASET_SCHEMA_VERSION,
    EpisodeSamples,
    NpzShardDataset,
    NpzShardWriter,
    deterministic_subsample_indices,
    discounted_returns,
    load_npz_shard,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EpisodeSamples",
    "NpzShardDataset",
    "NpzShardWriter",
    "deterministic_subsample_indices",
    "discounted_returns",
    "load_npz_shard",
]
