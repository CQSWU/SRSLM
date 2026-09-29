# SRSLM

This repository provides the implementation of **SRSLM: Switch and Reweight
with Shared Trace for Lifelong Partially Observable Multi-Agent Pathfinding**.
SRSLM combines the search-based AORePlan policy with the learned ARPE policy
and uses a Switcher to select between their actions. When AORePlan proposes
wait, SRSLM uses ARPE directly; otherwise, the Switcher chooses the branch.

| RePlan | AORePlan |
| :---: | :---: |
| <img src="docs/assets/replan.svg" alt="RePlan animation" width="420"> | <img src="docs/assets/aoreplan.svg" alt="AORePlan animation" width="420"> |

An illustrative two-agent example.

## Installation

Python 3.10 or 3.11 and a C++ compiler are required.

```bash
git clone https://github.com/CQSWU/SRSLM.git
cd SRSLM
uv sync
```

The planner extension is compiled automatically on first use.

## Inference Example

AORePlan does not require pretrained weights:

```bash
uv run python run_experiments.py \
  --algorithms AORePlan \
  --map-types wc3 --map wc3=wc3-128x64-TimbermawHold
```

The public methods are AORePlan, ARPE and SRSLM. Baseline and ablation entry
points are not included; EPOM-L is retained only as the learned methods' base.

For learned methods, place the separately supplied `weights/` directory in
the project root. Current settings are listed in
[CURRENT_VERSION.md](CURRENT_VERSION.md).
Use your own compatible weights by editing the paths in
`configs/arpe_final_candidate.json` and setting `--switcher-weights-path`.
No hash registration or audit manifest is required.

## Training

Configurable entry points; exact saved training configurations and curriculum
drivers are retained with the experiment backup:

```bash
# ARPE
uv run python train.py --config_path learning/train_arpe_final.yaml

# Switcher
uv run python train_switcher.py --config_path learning/train_switcher.yaml
```

Training maps are listed in `maps/train.yaml`, and evaluation maps are listed
in `maps/test.yaml`.
Training YAMLs are editable examples: choose your own maps, agent counts,
workers and PPO settings for your hardware.

ARPE loads the supplied EPOM-L backbone; standalone EPOM fine-tuning code is
not included. AORePlan builds on RePlan, and ARPE builds on EPOM; the original
authors' notices are preserved in [LICENSE](LICENSE).

## Documentation

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Current implementation](CURRENT_VERSION.md)
- [Map sets](docs/MAPS.md)

## Citation

If you use this repository, please cite:

```bibtex
@software{xie2026srslm,
  title  = {SRSLM: Switch and Reweight with Shared Trace for Lifelong Partially Observable Multi-Agent Pathfinding},
  author = {Jinghu Xie},
  year   = {2026},
  url    = {https://github.com/CQSWU/SRSLM}
}
```

## License

This project is released under the [MIT License](LICENSE).
