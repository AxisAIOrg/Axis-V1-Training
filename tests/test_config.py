from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from axis_train.config import load_config, read_json, write_json


def test_load_config_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"batch_size": 8, "seed": 42}), encoding="utf-8")
    assert load_config(path) == {"batch_size": 8, "seed": 42}


def test_load_config_json(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"lr": 1e-4}), encoding="utf-8")
    assert load_config(path) == {"lr": 1e-4}


def test_load_config_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "cfg.txt"
    path.write_text("x: 1", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config suffix"):
        load_config(path)


def test_write_and_read_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.json"
    write_json(path, {"ok": True, "n": 3})
    assert read_json(path) == {"ok": True, "n": 3}
