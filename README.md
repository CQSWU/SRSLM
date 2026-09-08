# SRSLM

| RePlan | AORePlan |
|:---:|:---:|
| ![RePlan animation](docs/assets/replan_demo.svg) | ![AORePlan animation](docs/assets/aoreplan_demo.svg) |

Reference implementation for **SRSLM: Switch and Reweight with Shared Trace
Memory for Lifelong Partially Observable Multi-Agent Pathfinding**.

## Current method

The public implementation contains the proposed method, its planning baseline,
and the ablations needed to inspect each component:

- **RePlan** is the original dynamic replanning baseline.
- **AORePlan** changes only a reverse move. It queries A* once on the static
  map and uses a usable non-reverse first step; a missing or locally
  conflicting static step becomes wait. The previous position is recorded at
  every timestep, including waits and blocked moves. RePlan's original
  BestMove and 50% random / 50% stay no-path fallback are otherwise preserved.
- **EPOM-L** is the lifelong fine-tuned recurrent base policy used by CAAR.
- **Direct** uses the frozen EPOM-L policy, not the older independent
  NoReweight model. It subtracts signed pressure sampled from the free-cell-
  mean-centred 11x11 trace. Policy entropy and capped ReLU are explicit,
  separately recorded options; the default is plain signed Direct.
- **CAAR** freezes EPOM-L and trains a separate trace branch. A Conv32 encoder
  with two residual blocks maps the mean-centred 11x11 trace to 32 features.
  These features are fused with the frozen 512-dimensional recurrent state and
  five base logits. The branch outputs five logit corrections. A policy-entropy
  gate decides whether to apply them; no action mask is an input to the learned
  branch. CAAR has 303,846 trainable parameters.
- **Switcher** is a feed-forward PPO policy with two outputs: choose CAAR or
  choose AORePlan. It does not output a primitive grid action.
- **SRSLM** uses CAAR immediately when AORePlan proposes wait. For every
  non-wait AORePlan proposal, Switcher samples one of the two complete
  candidate actions. Training and evaluation share the same routing code.
- **SRSLM-OnlyWait** is the deterministic ablation: use CAAR on a planner
  wait, otherwise use AORePlan. **SRSLM-NoWait** loads its independently
  trained all-state Switcher. Both use the same hash-pinned CAAR as SRSLM.

Historical value-estimator switchers and experimental compatibility branches
are not part of this source release.

The legacy `NoReweight` name still denotes the independently trained backbone
used in earlier experiments. It is not an alias for the paper's EPOM-L base.

## Installation

Python 3.10 or 3.11 and a C++ compiler are required. `cppimport` builds the
planner extension from `planning/planner.cpp` when it is first imported.

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
  --map-file maps/srlsm_smoke.map \
  --agents 16 --seeds 0 --workers 1 \
  --obs-radius 5 --max-steps 128 \
  --on-target restart --collision-system block_both \
  --output-dir results --output aoreplan_smoke.json
```

## Retained checkpoints versus retraining

To reproduce the retained paper models, use the checkpoint files and immutable
configs identified in [CURRENT_VERSION.md](CURRENT_VERSION.md), together with
the public candidate declarations. This is inference-only reproduction; do not
resume training in those artifact directories. Weights are distributed
separately from Git.

For CAAR, Full SRSLM, NoWait and OnlyWait, put the weight directories at their
declared paths relative to this source checkout. `--main-dir` must be this
checkout; it is not a separate artifact-only root for these learned methods.

Retraining is a different workflow. The following are the recorded training
recipes, not a claim that a new run will reproduce the old checkpoint bytes or
SHA-256 values. Use fresh run names/directories and complete each upstream
stage before updating the next stage's artifact declaration:

```bash
# EPOM lifelong fine-tune recipe (100M frames)
uv run python train.py \
  --config_path learning/train_epom_lifelong_finetune_r5_100m.yaml \
  --run_name EPOM-L-retrain --train_dir weights/retraining/EPOM-L

# CAAR recipe (500M frames), after setting epom_base_weights_path
# in a working copy of the YAML to the intended frozen EPOM-L run
uv run python train.py \
  --config_path learning/train_epom_trace_paper_conv_fusion_r5_500m.yaml \
  --run_name CAAR-retrain --train_dir weights/retraining/CAAR

# Switcher recipe, after updating its candidate_policy declaration
uv run python train_switcher_wait_caar.py \
  --config_path learning/train_switcher_wait_caar_100m_server2.yaml \
  --run_name Switcher-retrain --train_dir weights/retraining/Switcher
```

The unmodified CAAR YAML's run name ends in `-500M`; the retained candidate
declaration instead names the historical `-S0-20260902` run. A new run must not
be relabelled as that old artifact. After training, create a new declaration
from `configs/caar_final_candidate.json` with the actual relative CAAR/base
paths and SHA-256 values of both checkpoint files and both config files. Put
that same declaration in the working Switcher YAML's `candidate_policy`, and
match `environment.switcher_caar_weights_path` to it. Keep the new declaration
separate from the retained paper declaration. NoWait uses the analogous
`train_switcher_nowait.py` entrypoint and
`learning/train_switcher_nowait_caar_100m_server1.yaml`; it is independently
trained, not a Full checkpoint with the wait rule disabled. OnlyWait has no
trainable Switcher and needs no separate training stage.
The generic `train.py` environment factory rejects both Switcher environment
names; use the dedicated entrypoint so the candidate declaration is enforced.

Use the corresponding `*_smoke.yaml` files before a full run. The shell
launchers under `scripts/` add PPU allocation, duplicate-run protection,
checkpoint hashing, and postflight validation for the audited server setup.

## Exact960 protocol

The retained formal result uses 32 held-out maps, populations
100/200/300/400/500/600, seeds 0/42/123/2024/3407, `block_both` collisions,
lifelong `restart`, 512 steps, and observation radius 5. This is exactly 960
map-population-seed episodes.

The validated SRSLM run contains all 960 unique finite error-free rows and has
mean throughput **1.8608784993**. The complete artifact identities and
per-population values are recorded in [CURRENT_VERSION.md](CURRENT_VERSION.md).

With the hash-pinned checkpoints in their documented paths, a fresh evaluation is:

```bash
uv run python run_experiments.py \
  --algorithms SRSLM --map-list maps/eval_capacity_intersection_n600.yaml \
  --agents 100,200,300,400,500,600 --seeds 0,42,123,2024,3407 \
  --on-target restart --collision-system block_both \
  --obs-radius 5 --max-steps 512 --workers 1 \
  --output-dir results/srslm_block --output srslm.json
```

Use a new output directory for each arm and choose workers to fit available
memory. The retained server launcher
`scripts/run_srslm_wait_aware_caar_100m_exact960_server1.sh` additionally checks
the original training-validation and readiness artifacts. It requires that
separate audit package, not just a fresh source checkout. Its `PROJECT_ROOT`
can be set to the checkout path; those historical checks have not been removed.

### Soft evaluation and the search-only exception

Use `--collision-system soft` with the same block-trained weights for the
paper's soft evaluation. Default AORePlan, including the planner inside
SRSLM and its switching ablations, still converts a static A* step into wait
when its destination is already occupied in the current observation.

Only the standalone **search ablation** omits this extra check under soft:
use `--algorithms RePlan,AORePlan-SoftNoCheck --collision-system soft`.
The explicit name is retained in result rows so those data cannot be mistaken
for the default AORePlan implementation. This option does not change SRSLM.

### CAAR and Direct ablations

The public, path-relative candidate declaration is
`configs/caar_final_candidate.json`. For `CAAR`, `SRSLM-OnlyWait`, or
`SRSLM-NoWait`, pass `--caar-candidate-manifest configs/caar_final_candidate.json`.
NoWait additionally requires its independently trained
`--switcher-weights-path`; OnlyWait must not receive Switcher weights.

The independently trained **CAAR without policy entropy** is evaluated with
`--algorithms CAAR --caar-candidate-manifest configs/caar_noentropy_candidate.json`
in its own output file. This loads the separate 500M-step ungated checkpoint;
it is not an inference-time gate override of the gated CAAR. Its retraining
recipe is `learning/train_caar_noentropy_r5_500m.yaml`, which changes only the
training gate to `all` and uses a separate run name/output directory. The
backbone, architecture, PPO settings, maps, population and frame budget match
the gated recipe. A fresh run must receive its own checkpoint declaration and
hashes, not the retained ungated artifact's hashes.

All Direct arms require `--epom-weights-path` pointing to the EPOM-L directory
listed in [CURRENT_VERSION.md](CURRENT_VERSION.md):

| Arm | Arguments in addition to `--algorithms Direct` |
| --- | --- |
| Direct | `--direct-gate always --direct-centering crop --direct-transform signed` |
| Direct + policy entropy | `--direct-gate primal3 --direct-centering crop --direct-transform signed` |
| Direct + entropy + capped ReLU | `--direct-gate primal3 --direct-centering crop --direct-transform clipped_relu` |

Run these arms in separate output files. Both metadata and model provenance
record the gate, centring scope, and transform. Capped ReLU means
`min(max(P, 0), 2)` after centring. Learned CAAR remains an unclipped correction;
these Direct options do not modify it. `--direct-centering candidate` is only
an explicit historical five-cell ablation, not the current paper default.

External DCC-L provenance and redistribution limits are documented in
[DCC-L reproduction notes](docs/DCC_L_REPRODUCTION.md). Its implementation and
weights are not vendored into this public source tree.

## Tests

```bash
uv run python -m pytest \
  tests/test_ao_replan.py \
  tests/test_aoreplan_branch.py \
  tests/test_direct.py \
  tests/test_epom_direct_reweight.py \
  tests/test_epom_paper_entropy_fusion.py \
  tests/test_switcher.py \
  tests/test_switcher_learner_patch.py \
  tests/test_switcher_caar_candidate.py \
  tests/test_srslm_candidate_binding.py \
  tests/test_srslm_caar_ablation.py \
  tests/test_public_training_contracts.py \
  tests/test_switcher_training_entrypoints.py \
  tests/test_runner_current_contracts.py
```

See [CURRENT_VERSION.md](CURRENT_VERSION.md) for the frozen artifact hashes and
[DIRECTORY.md](DIRECTORY.md) for the intentionally small public source layout.

## License

MIT. See [LICENSE](LICENSE).
