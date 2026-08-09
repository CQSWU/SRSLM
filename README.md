# SRSLM

Reference implementation for **SRSLM: Switch and Reweight with Shared Trace
Memory for Lifelong Partially Observable Multi-Agent Pathfinding**.

This release intentionally contains only the three methods used by SRSLM:

- **AO-RePlan** - the action-observation planning baseline.
- **CAAR** - a recurrent MAPF policy whose action logits are reweighted
  with a pressure signal derived from the shared, decaying traffic trace.
- **SRSLM** - CAAR and raw AO-RePlan proposals are compared at every
  step using two independent absolute-return estimators. A reverse AO proposal
  is rejected for that timestep and replaced by the CAAR action. The next
  timestep starts with a new value comparison.

The repository includes the map registries, source code, training entry points,
and focused tests. Released CAAR and SRSLM weights are available from the
[SRSLM v1.0.0 release](https://github.com/CQSWU/SRSLM/releases/tag/v1.0.0).
Historical experiments, external baselines, and obsolete switchers are not
included.

## Installation

Python 3.10 or 3.11 and a C++ compiler are required. `cppimport` builds the
planner extension from `planning/planner.cpp` on first use.

```bash
git clone https://github.com/CQSWU/SRSLM.git
cd SRSLM
uv sync --extra test
```

If `uv` is unavailable, create a Python 3.10/3.11 environment and install the
locked project dependencies with your preferred package manager.

## Quick AO-RePlan smoke run

AO-RePlan does not require a learned checkpoint:

```bash
uv run python run_experiments.py \
  --algorithms AO-RePlan \
  --map-file maps/srlsm_smoke.map \
  --agents 16 --seeds 0 --workers 1 \
  --max-steps 128 --on-target finish \
  --output-dir results --output ao_replan_smoke.json
```

## Train and evaluate CAAR

The short configuration is for a functionality check. The R5 configuration is
the full training recipe; it is not a quick smoke run.

```bash
# Functionality smoke training
uv run python train_caar.py \
  --config_path learning/train_caar_r5_smoke.yaml

# Full CAAR R5 recipe
uv run python train_caar.py \
  --config_path learning/train_caar_r5.yaml
```

After the full run, evaluate the resulting checkpoint directory:

```bash
uv run python run_experiments.py \
  --algorithms CAAR \
  --caar-weights-path weights/CAAR/radius_ablation/R5 \
  --map-list maps/eval.yaml \
  --agents 100,150,200,250,300,350,400 \
  --seeds 0,42,123,456,789 --workers 16 \
  --max-steps 512 --on-target restart \
  --output-dir results --output caar_lifelong.json
```

## Train and evaluate SRSLM

SRSLM requires a frozen CAAR checkpoint and two independently trained
return estimators. The collection script writes paired CAAR/AO-safe trajectories
from identical scenarios; the trainer then writes `caar_estimator.pth` and
`ao_estimator.pth` plus a provenance manifest.

```bash
# Collect paired trajectories. Replace the small example scenario set with a
# formal train/validation scenario manifest for a paper-scale run.
uv run python scripts/collect_caar_ao_returns.py \
  --scenario-manifest configs/trace_smoke_scenarios.yaml \
  --output data/trace_smoke \
  --caar-weights weights/CAAR/radius_ablation/R5 \
  --sample-fraction 1.0 --workers 1

# Train the two value estimators.
uv run python scripts/train_caar_ao_estimators.py \
  --data data/trace_smoke/caar data/trace_smoke/ao_safe \
  --output weights/SRSLM-v1 \
  --num-trials 1 --epochs-per-trial 1 --overfit-small-data

# Run the learned, rule-constrained switcher.
uv run python run_experiments.py \
  --algorithms SRSLM \
  --caar-weights-path weights/CAAR/radius_ablation/R5 \
  --srlsm-caar-estimator-checkpoint weights/SRSLM-v1/caar_estimator.pth \
  --srlsm-ao-estimator-checkpoint weights/SRSLM-v1/ao_estimator.pth \
  --map-file maps/srlsm_smoke.map \
  --agents 16 --seeds 0 --workers 1 --max-steps 128 \
  --output-dir results --output trace_srlsm_smoke.json
```

The small scenario set is only a pipeline check. Reproducing paper-scale
numbers requires the full map/seed protocol and the corresponding training
budget; it must not be interpreted as a numerical reproduction of the paper.

## Tests

```bash
uv run python -m unittest discover -s tests -p test_ao_replan.py
uv run python -m unittest discover -s tests -p test_raw_aoreplan_candidates.py
uv run python -m unittest discover -s tests -p test_trace_switcher_contract.py
uv run python -m unittest discover -s tests -p test_caar_pe_model.py
uv run python -m unittest discover -s tests -p test_caar_ao_rollout.py
uv run python -m unittest discover -s tests -p test_stigmergic.py
```

## Release boundary and reproducibility

The released core algorithm modules match the runtime used for the retained
SRSLM formal results: `run_experiments.py`, `agents/caar.py`,
`agents/srlsm.py`, the encoder, planner, estimator, and environment.
The public `uv.lock` is regenerated for a clean installation and is therefore
treated as a separate, versioned runtime artifact. SHA-256 values are recorded
in result/provenance files for auditability only: normal deployment does not
reject a checkpoint merely because the user changed source code or a runtime
parameter. Incompatible checkpoint schemas or tensor shapes still fail clearly.

Pretrained checkpoints are intentionally not committed to Git. They are
published as versioned GitHub Release assets with SHA-256 checksums rather than
silently replacing files under `weights/`.

## License

MIT. See [LICENSE](LICENSE).
