# Reproducibility

Selected inference settings and checkpoint folders are listed in
[CURRENT_VERSION.md](../CURRENT_VERSION.md). Weights and raw experiment records
are kept separately from Git. Compatible custom weights can be loaded without
editing a hash allowlist.

Public runs do not require a paper audit, source-tree hash check, or a match
between the saved training protocol and the new evaluation protocol. Training
YAMLs are examples, not enforced recipes: maps, populations, episode length,
worker count and PPO parameters can be changed. Network shapes and action
encoding must still be compatible with the checkpoint. The frozen branches
remain frozen; removing audit checks does not change their numerical behavior.

## Inference

Copy `EPOM-L`, `ARPE-Final-1B` and `SRSLM-Switcher-Final-1B` from the supplied
weight bundle into `weights/`. Preserve each folder's `config.json` and
`checkpoint_p0/` files. Run the evaluator from the repository root.

AORePlan has no neural checkpoint. It obtains RePlan's proposal, including the
original greedy/no-path fallback, and checks whether it returns to the previous
timestep's position. A reverse proposal triggers a query on the accumulated
static map. A blocked static proposal becomes wait. Every timestep records the
position, including waits. Successful movement releases failure caches; new
failed destinations are cached with probability 0.5.

Only AORePlan, ARPE and SRSLM are public evaluator entries. Historical
baseline and ablation sources are kept in the separate experiment archive.

Current ARPE inference disables Direct and applies a x12 learned residual
only when base-policy entropy exceeds 0.01. Full SRSLM uses ARPE for AORePlan
waits, and the Switcher for every non-wait proposal, including reverse moves.
Original training configs must not be mistaken for these inference settings.

Standalone ARPE and the current SRSLM learning branch retain the historical
Direct-style NumPy action sampler; Switcher retains its stochastic sampler.
Trace, recurrent memory, grid memory and
planner state must be reset at episode boundaries. Lifelong goal replacement
is not a new episode.

## Data and maps

Training and test maps are consolidated in `maps/train.yaml` and
`maps/test.yaml`. The latter contains the original 32 maps and four resized
MovingAI WC3 maps; see [MAPS.md](MAPS.md).

Full evaluations use all 36 maps, populations 100/200/300/400/500/600, seeds
0/42/123/2024/3407, 512 steps, radius 5 and lifelong `restart`. This gives
1,080 unique episodes per method and collision setting. `block_both` and `soft`
are separate evaluations. Recorded soft results use the original simulator
behavior, not the later occupancy fix.

Experiment results and data backups are kept locally, not in this repository.
Keep raw rows unchanged. Decision percentages use pooled counts.
New SRSLM runs count only wait overrides. Older two-rule runs retain
their original wait and reverse counts and must not be relabeled as wait-only.
Long-episode analysis retains all eight 512-step throughput windows rather than
only an overall mean. Old 960/1050 runs and incomplete historical records should
not be silently substituted for full 1080 results.

## Training and tests

`train.py` and `train_switcher.py` are configurable training entry points.
Exact saved configurations, continuation state and curriculum drivers belong
to the original run archive. Example YAML files are starting points, not a
promise to recreate a stochastic checkpoint bit for bit. In particular, the
final ARPE inference gate/multiplier was applied after training, and the final
Switcher was evaluated with a replacement frozen ARPE candidate.

Run `python -m pytest -q` for the portable regression suite. These tests check
code behavior; they do not validate unprovided historical data. Publication
audits live in the result bundle and do not impose extra execution gates.

External baseline implementations and weights are not vendored into the public
repository; their upstream licenses and dependencies still apply.
