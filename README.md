# SRSLM

Reference implementation for **SRSLM: Switch and Reweight with Shared Trace
for Lifelong Partially Observable Multi-Agent Pathfinding**.

The public repository contains the current proposed method, the planning
baseline, and the ablations needed to inspect each component. Model weights,
raw experiment outputs, private server environments, and third-party baseline
repositories are distributed separately and are not committed here.

## Current method

- **RePlan** is the original dynamic replanning baseline.
- **AORePlan** checks a reverse proposal with A* on the accumulated static map.
  A missing or locally conflicting static first step becomes wait; the original
  RePlan no-path fallback is otherwise preserved. The planner also contains the
  current cache-release and randomized failure-caching fixes.
- **EPOM-L** is the lifelong fine-tuned recurrent base policy used by Direct and
  ARPE.
- **Direct** applies a parameter-free shared-trace correction to frozen EPOM-L.
  Its entropy gate, centring scope, and pressure transform are explicit in each
  run's configuration and provenance.
- **ARPE** freezes EPOM-L and trains a separate trace branch. The branch encodes
  the mean-centred 11x11 shared trace, fuses it with the frozen recurrent state
  and base logits, and predicts five logit corrections.
- **Switcher** is a two-action categorical policy. It selects the complete ARPE
  or AORePlan proposal rather than a primitive grid move.
- **SRSLM** directly uses ARPE when AORePlan proposes wait. Otherwise, Switcher
  samples between the two complete proposals. Training and inference use the
  same routing implementation.

`SRSLM-OnlyWait` and `SRSLM-NoWait` are retained switching ablations. Historical
value-estimator switchers, NoReweight, learned Follower branches, SRSLMF, and
other retired compatibility paths are not part of the public implementation.

## Installation

Python 3.10 or 3.11 and a C++ compiler are required. `cppimport` builds the
planner extension from `planning/planner.cpp` on first use.

```bash
git clone https://github.com/CQSWU/SRSLM.git
cd SRSLM
uv sync --extra test
```

## Quick AORePlan check

AORePlan has no learned parameters:

```bash
uv run python run_experiments.py \
  --algorithms AORePlan \
  --map-types wc3 --map wc3=wc3-128x64-TimbermawHold \
  --agents 16 --seeds 0 --workers 1 \
  --obs-radius 5 --max-steps 128 \
  --on-target restart --collision-system block_both \
  --output-dir results --output aoreplan_smoke.json
```

## Checkpoints and retraining

The selected artifact identities and hashes are listed in
[CURRENT_VERSION.md](CURRENT_VERSION.md). Weights are not downloaded by the
source checkout. Put them at the declared relative paths and verify every hash
before evaluation.

The retained training recipes are:

```text
learning/train_epom.yaml
learning/train_arpe.yaml
learning/train_switcher.yaml
```

These files are reproducible recipes, not claims that a fresh run will recreate
historical checkpoint bytes. Use a new output directory and a new artifact
declaration for every retraining run. Train EPOM-L and ARPE with `train.py`.
Train the final Switcher with `train_switcher.py`, which verifies the pinned
candidate-policy hashes and routing contract before starting Sample Factory.

## Evaluation protocol

The current paper grid contains 36 held-out maps: the original 32-map
capacity-intersection set plus four resized MovingAI WC3 maps. It uses
populations 100/200/300/400/500/600, seeds 0/42/123/2024/3407, lifelong
`restart`, 512 steps, and observation radius 5. This gives 1,080 unique
map-population-seed episodes per method and execution rule.

The repository has one training registry and one test registry:

```text
maps/train.yaml
maps/test.yaml
```

`test.yaml` already contains the original 32 held-out maps and the four
resized MovingAI WC3 maps; no secondary list or registry is required.

Run `block_both` and `soft` separately with the same selected block-trained
weights. Keep each output directory immutable and require unique, finite,
error-free rows together with source, map, configuration, and checkpoint hashes
before using a result in the paper. The older 32-map/960-row bundles remain
historical evidence and must not be relabelled as the 36-map result.

Default AORePlan keeps its conservative local occupancy check under both
execution rules, including inside SRSLM. Only the standalone
`AORePlan-SoftNoCheck` search ablation omits that check under `soft`; its name is
recorded explicitly so it cannot be confused with the deployed planner.

## Tests

```bash
uv run python -m pytest tests
```

The focused regression suite covers AORePlan cache release, accumulated static
memory, collision handling, trace reset across episodes, ARPE/Switcher artifact
contracts, and the restricted public method registry.

## License

MIT for this project's source. Third-party projects and released checkpoints
retain their own licenses.
