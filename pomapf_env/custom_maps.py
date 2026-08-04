from pathlib import Path


import yaml


_root = Path(__file__).parent.parent / "maps"

maps = {}

for _fname in ("train.yaml", "eval.yaml"):

    _path = _root / _fname

    if _path.exists():

        with open(_path, "r") as f:

            maps.update(yaml.safe_load(f))


MAPS_REGISTRY = maps

