from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from axis_train.config import write_json
from axis_train.data.lerobot_v3 import LeRobotV3Dataset, format_episode_id


def make_episode_split(
    dataset_path: str | Path,
    *,
    output_path: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    id_format: str = "int",
) -> dict[str, Any]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios < 0):
        raise ValueError(f"Split ratios must be finite and non-negative, got {ratios.tolist()}")
    if not np.isclose(float(ratios.sum()), 1.0, atol=1.0e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {float(ratios.sum())}")
    if train_ratio <= 0:
        raise ValueError("train_ratio must be positive")

    dataset = LeRobotV3Dataset.from_path(dataset_path)
    episodes = dataset.read_episodes_table()
    episode_ids = np.asarray(episodes["episode_index"].to_pylist(), dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total == 0:
        raise ValueError("No episodes available for split")
    n_train = max(1, int(round(total * train_ratio)))
    n_val = int(round(total * val_ratio))
    if n_train + n_val > total:
        n_val = max(0, total - n_train)
    n_test = total - n_train - n_val
    if val_ratio > 0 and n_val == 0 and n_train > 1:
        n_train -= 1
        n_val = 1
    if test_ratio > 0 and n_test == 0 and n_train > 1:
        n_train -= 1
        n_test = 1

    split_ids = {
        "train": shuffled[:n_train].tolist(),
        "val_seen": shuffled[n_train : n_train + n_val].tolist(),
        "test_seen": shuffled[n_train + n_val :].tolist(),
        "test_unseen_init": [],
        "test_unseen_visual": [],
        "test_unseen_object": [],
        "test_long_horizon": [],
    }
    if id_format == "episode":
        payload = {k: [format_episode_id(int(x)) for x in v] for k, v in split_ids.items()}
    else:
        payload = {k: [int(x) for x in v] for k, v in split_ids.items()}
    payload["_meta"] = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "seed": seed,
        "num_episodes": int(total),
        "ratios": {"train": train_ratio, "val_seen": val_ratio, "test_seen": test_ratio},
    }
    write_json(output_path, payload)
    return payload
