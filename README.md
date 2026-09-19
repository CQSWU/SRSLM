# SRSLM

This repository provides the implementation of **SRSLM: Switch and Reweight
with Shared Trace for Lifelong Partially Observable Multi-Agent Pathfinding**.
SRSLM combines the search-based AORePlan policy with the learned ARPE policy
and uses a Switcher to select between their actions.

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
  --algorithms RePlan,AORePlan \
  --map-types wc3 --map wc3=wc3-128x64-TimbermawHold
```

For EPOM-L, Direct, ARPE, and SRSLM, place the released `weights/` directory
in the project root before running the evaluator.
Custom compatible weights are supported; recorded hashes do not restrict loading.

## Training

```bash
# EPOM-L
uv run python train.py --config_path learning/train_epom.yaml

# ARPE
uv run python train.py --config_path learning/train_arpe.yaml

# Switcher
uv run python train_switcher.py --config_path learning/train_switcher.yaml
```

Training maps are listed in `maps/train.yaml`, and evaluation maps are listed
in `maps/test.yaml`.

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
