# SRSLM

**Switch and Reweight with Shared Trace for Lifelong Partially Observable
Multi-Agent Pathfinding.**

SRSLM combines AORePlan and ARPE with a learned Switcher. When AORePlan
returns wait, ARPE is used directly; otherwise, the Switcher selects a branch.

| RePlan | AORePlan |
| :---: | :---: |
| <img src="docs/assets/replan.svg" alt="RePlan animation" width="420"> | <img src="docs/assets/aoreplan.svg" alt="AORePlan animation" width="420"> |

## Install

Linux, Python 3.10 or 3.11, and a C++ compiler:

```bash
git clone https://github.com/CQSWU/SRSLM.git
cd SRSLM
uv sync
```

## Run

```bash
uv run python run_experiments.py --algorithms AORePlan
```

For ARPE and SRSLM, put the supplied `EPOM-L`, `ARPE-Final-1B` and
`SRSLM-Switcher-Final-1B` folders in `weights/`, keeping their `config.json`
and checkpoint files. Edit `configs/arpe_final_candidate.json` to change ARPE
weight paths or inference settings; use `--switcher-weights-path` for Switcher.

```bash
uv run python run_experiments.py --algorithms SRSLM
```

ARPE uses the learned correction at 12 times its training scale when base-policy
entropy exceeds 0.01, without Direct's bonus. The supplied ARPE was trained with
Direct bonus 1 and always-on correction. The supplied Switcher was trained with
the earlier ARPE and wait/reverse rules; it has not been retrained for the current
ARPE replacement and wait-only rule.

## Train

```bash
uv run python train.py --config_path learning/train_arpe_final.yaml
uv run python train_switcher.py --config_path learning/train_switcher.yaml
```

Edit the YAML files to set maps, populations, workers and PPO parameters.
They are fresh-run examples, not the original multi-stage training schedule.
Weights and experiment results are distributed separately from the source.

`maps/train.yaml` contains 186 training maps; `maps/test.yaml` contains 36 test
maps. Four [MovingAI WC3 maps](https://www.movingai.com/benchmarks/wc3maps512/index.html)
were reduced from 512×512 to 128×64 using 4×8 blocks, marking a cell free when
at least half its source cells are free.

## License

[MIT](LICENSE). AORePlan builds on RePlan, and ARPE uses EPOM as its base policy;
the original authors' notices are preserved in the license.
