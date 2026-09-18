from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _registry(name):
    return yaml.safe_load((ROOT / "maps" / name).read_text(encoding="utf-8"))


def test_project_has_one_training_and_one_test_registry():
    yaml_files = sorted(path.name for path in (ROOT / "maps").glob("*.yaml"))
    assert yaml_files == ["test.yaml", "train.yaml"]

    train = _registry("train.yaml")
    test = _registry("test.yaml")
    assert len(train) == 186
    assert len(test) == 36
    assert all(isinstance(grid, str) and grid.strip() for grid in train.values())
    assert all(isinstance(grid, str) and grid.strip() for grid in test.values())


def test_resized_wc3_extension_is_inside_the_test_registry():
    test = _registry("test.yaml")
    assert {
        "wc3-128x64-DustwallowKeys",
        "wc3-128x64-Icecrown",
        "wc3-128x64-PlunderIsle",
        "wc3-128x64-TimbermawHold",
    } <= set(test)
