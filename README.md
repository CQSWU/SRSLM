# SRSLM

Reference implementation for **SRSLM: Switch and Reweight with Shared Trace
Memory for Lifelong Partially Observable Multi-Agent Pathfinding**.

The selected reweighting method is now **ARPE (Action Reweight with Policy
Entropy)**, previously called CAAR. This is a naming change, not a new model.
See [method naming](docs/METHOD_NAMING.md) for the few saved-checkpoint
identifiers and historical result names that remain unchanged.

For the current paper's exact data, checkpoints, figure sources and known
provenance gaps, start with [PAPER_REPRODUCIBILITY.md](docs/PAPER_REPRODUCIBILITY.md).
Retired model architectures are no longer runnable in the main source tree.
The pre-cleanup source is sealed in `artifacts/reproducibility/`, not mixed
with the current implementation.

## Retained methods

- **RePlan** is the original search baseline.
- **AORePlan** checks reverse proposals with static-map A*. It otherwise keeps
  RePlan's planner and no-path fallback.
- **Direct** applies signed shared-trace correction to frozen EPOM-L. It centres
  over free cells of the full 11x11 crop; entropy gating and clipped ReLU
  are explicit options, not implicit changes of the base policy.
- **ARPE** learns a five-action trace correction on a frozen EPOM-L base.
- **Switcher** is a two-action categorical policy: it selects ARPE or
  AORePlan, not a primitive grid move.
- **SRSLM** uses ARPE immediately when AORePlan returns wait; every other state
  is sent to Switcher.

The internal runner also retains DCC/DCC-L, EPOM/EPOM-L, PRIMAL2, CHS, and the official FoLLow
adapter. FoLLow is opt-in because its released configuration uses `priority`
collisions. The main comparisons report `block_both` and `soft` separately.

The public source release has a smaller method registry. Internal external
adapters and their separately licensed source/checkpoint packages are not
supplied by cloning the public repository alone. DCC-L uses the selected
u100 checkpoint documented in [DCC_L_REPRODUCTION.md](docs/DCC_L_REPRODUCTION.md),
not a stale default checkpoint. This internal source sync does not install
or retrain any external method.

## Installation

Python 3.10 or 3.11 and a C++ compiler are required. `cppimport` builds the
planner extension from `planning/planner.cpp` on first use.
The commands below describe a fresh public-source checkout, not an upgrade
of an existing server environment. Do not run `uv sync` against an active
PPU experiment: retain its installed PPU SDK, environment and native binaries.

```bash
git clone https://github.com/CQSWU/SRSLM.git
cd SRSLM
uv sync --extra test
```

## Quick check

```bash
uv run python run_experiments.py \
  --algorithms AORePlan \
  --map-file maps/srlsm_smoke.map \
  --agents 16 --seeds 0 --workers 1 \
  --obs-radius 5 --max-steps 128 \
  --on-target restart --collision-system block_both \
  --output-dir results --output aoreplan_smoke.json
```

The selected ARPE checkpoint is pinned by
`artifacts/arpe_final_candidate.json`. The retained Switcher configurations
are:

```text
learning/train_switcher_wait_arpe_100m_server2.yaml
learning/train_switcher_nowait_arpe_100m_server1.yaml
```

Their launchers are
`scripts/run_train_switcher_wait_arpe_server2.sh` and
`scripts/run_train_switcher_nowait_arpe_server1.sh`.

## Current inference and training contracts

Full SRSLM, NoWait, and OnlyWait use the same hash-pinned ARPE. NoWait uses
its independently trained Switcher; OnlyWait has no Switcher checkpoint.
Use `configs/arpe_final_candidate.json` for the selected ARPE. The declaration
identifies required artifacts; it does not download them.
See the server artifact availability note in
[CURRENT_VERSION.md](CURRENT_VERSION.md) before using either declaration.

For ARPE/Full/NoWait/OnlyWait, all declared artifact paths are relative to
this source checkout. `--main-dir` is not a separate weights-only root for
these methods. An explicit EPOM-L weights path is relative to `--main-dir`
unless already absolute. Direct requires `--epom-weights-path` and defaults
to `--direct-gate always --direct-centering crop --direct-transform signed`;
use `primal3` for entropy gating and `clipped_relu` for the capped variant.

The default AORePlan keeps the conservative static-step occupancy check in
both collision modes, including inside SRSLM. Only the standalone soft
search ablation selects `AORePlan-SoftNoCheck`; it must not replace the
planner inside the retained soft SRSLM result.

Exact checkpoint reproduction and retraining are different tasks. The
current YAML files describe training recipes, not a promise to reproduce
historical checkpoint bytes or hashes. Use fresh run directories. Train
ARPE via `train.py` and use the dedicated `train_switcher_wait_arpe.py` or
`train_switcher_nowait.py` entrypoint for Switcher training; the generic
training entrypoint deliberately rejects Switcher environments. After a
new ARPE run, create a new candidate declaration with its actual base/model
paths and hashes and use it consistently in the next Switcher stage.

## Paper protocol

The main exact960 grid contains 32 maps, populations
100/200/300/400/500/600, seeds 0/42/123/2024/3407, lifelong `restart`, 512
steps and radius 5, evaluated separately under `block_both` and `soft`.
Use the same selected block-trained EPOM-L, ARPE and Switcher weights under
both execution rules; do not substitute a soft-trained model. The main table
contains DCC-L, EPOM-L, SRSLM, PRIMAL2 and CHS. PIBT is no longer a retained
comparison method. The standalone soft AORePlan ablation
uses the explicitly named exception described above.

DCC-L is the selected update-100 lifelong fine-tune, not the unmodified
official DCC checkpoint. Pass its verified location with
`--dcc-weights-path`; the default DCC path is not a DCC-L alias. See
[CURRENT_VERSION.md](CURRENT_VERSION.md) for existing server locations and
[DCC_L_REPRODUCTION.md](docs/DCC_L_REPRODUCTION.md) for the selection evidence.

Require the complete 960 unique, finite, error-free rows, exact protocol,
checkpoint/source identities and a checked validation record before using a
result. Historical results keep their original source manifests and known
provenance limitations; newer source cannot retroactively certify them.

FoLLow must be evaluated separately with `--collision-system priority`; its scores
must not be inserted into the shared `block_both`/`soft` table without a protocol
label.

See [CURRENT_VERSION.md](CURRENT_VERSION.md), [DIRECTORY.md](DIRECTORY.md), and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact retained
implementation and artifacts.

## Tests

On Windows/MSVC, set `CL=/utf-8 /permissive- /std:c++17` for the original
C++ planner's UTF-8 headers and standard `and`/`or` operators.

```bash
uv run python -m pytest tests/test_ao_replan.py tests/test_aoreplan_branch.py \
  tests/test_epom_paper_entropy_fusion.py tests/test_switcher.py \
  tests/test_srslm_arpe_ablation.py tests/test_follower_checkpoint_selection.py
```

## License

MIT for the project source. Third-party projects and released checkpoints keep
their original licenses; see [LICENSE](LICENSE).
