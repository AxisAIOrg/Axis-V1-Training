from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from axis_train.config import ensure_dir


@dataclass(frozen=True)
class LeRobotV3Dataset:
    path: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "LeRobotV3Dataset":
        return cls(Path(path))

    @property
    def info_path(self) -> Path:
        return self.path / "meta" / "info.json"

    @property
    def stats_path(self) -> Path:
        return self.path / "meta" / "stats.json"

    @property
    def tasks_path(self) -> Path:
        return self.path / "meta" / "tasks.parquet"

    @property
    def episodes_dir(self) -> Path:
        return self.path / "meta" / "episodes"

    @property
    def data_dir(self) -> Path:
        return self.path / "data"

    @property
    def videos_dir(self) -> Path:
        return self.path / "videos"

    def load_info(self) -> dict[str, Any]:
        with self.info_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_stats(self) -> dict[str, Any]:
        if not self.stats_path.exists():
            return {}
        with self.stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def iter_data_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("chunk-*/file-*.parquet"))

    def iter_episode_files(self) -> list[Path]:
        return sorted(self.episodes_dir.glob("chunk-*/file-*.parquet"))

    def read_data_table(self):
        files = self.iter_data_files()
        if not files:
            raise FileNotFoundError(f"No parquet data files under {self.data_dir}")
        return pq.read_table(files)

    def read_episodes_table(self):
        files = self.iter_episode_files()
        if not files:
            raise FileNotFoundError(f"No episode parquet files under {self.episodes_dir}")
        return pq.read_table(files)

    def read_tasks(self) -> dict[int, str]:
        if not self.tasks_path.exists():
            return {}
        table = pq.read_table(self.tasks_path)
        if "task_index" not in table.column_names or "task" not in table.column_names:
            return {}
        return {
            int(idx): str(task)
            for idx, task in zip(
                table["task_index"].to_pylist(),
                table["task"].to_pylist(),
                strict=True,
            )
        }

    def video_keys(self) -> list[str]:
        info = self.load_info()
        features = info.get("features", {})
        return [name for name, spec in features.items() if spec.get("dtype") == "video"]


def column_to_numpy(table, column: str, dtype=np.float32) -> np.ndarray:
    if column not in table.column_names:
        raise KeyError(f"Missing column {column}")
    return np.asarray(table[column].to_pylist(), dtype=dtype)


def scalar_column(table, column: str) -> np.ndarray:
    if column not in table.column_names:
        raise KeyError(f"Missing column {column}")
    return np.asarray(table[column].to_pylist())


def compute_array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.ndim == 1:
        flat = values.reshape(-1, 1)
    else:
        flat = values.reshape(values.shape[0], -1)
    return {
        "min": np.min(flat, axis=0).tolist(),
        "max": np.max(flat, axis=0).tolist(),
        "mean": np.mean(flat, axis=0).tolist(),
        "std": np.std(flat, axis=0).tolist(),
        "count": [int(flat.shape[0])],
        "q01": np.quantile(flat, 0.01, axis=0).tolist(),
        "q10": np.quantile(flat, 0.10, axis=0).tolist(),
        "q50": np.quantile(flat, 0.50, axis=0).tolist(),
        "q90": np.quantile(flat, 0.90, axis=0).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).tolist(),
    }


def numeric_feature_dims(info: dict[str, Any]) -> dict[str, int]:
    dims: dict[str, int] = {}
    for key, spec in info.get("features", {}).items():
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if dtype == "video" or not shape:
            continue
        if len(shape) == 1:
            dims[key] = int(shape[0])
    return dims


def episode_indices(dataset: LeRobotV3Dataset) -> list[int]:
    episodes = dataset.read_episodes_table()
    if "episode_index" not in episodes.column_names:
        raise KeyError("Episode metadata is missing episode_index")
    return [int(x) for x in episodes["episode_index"].to_pylist()]


def copy_tree_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    ensure_dir(dst.parent)
    if symlink:
        dst.symlink_to(os.path.abspath(src), target_is_directory=True)
    else:
        shutil.copytree(src, dst, symlinks=True)


def format_episode_id(index: int) -> str:
    return f"episode_{index:06d}"


def normalize_episode_ids(values: Iterable[int | str]) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, int):
            out.append(value)
            continue
        text = str(value)
        if text.startswith("episode_"):
            out.append(int(text.split("_")[-1]))
        else:
            out.append(int(text))
    return out
