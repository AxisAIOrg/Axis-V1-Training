from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from axis_train.config import ensure_dir, load_config, write_json
from axis_train.data.lerobot_v3 import (
    LeRobotV3Dataset,
    column_to_numpy,
    compute_array_stats,
    copy_tree_or_symlink,
)


@dataclass(frozen=True)
class ConversionResult:
    actions_8d: np.ndarray
    raw_width: np.ndarray
    scalar_gripper_unclipped: np.ndarray
    scalar_gripper: np.ndarray
    open_width: float
    closed_width: float
    clip_rate: float


def _resolve_quantile(value: Any, raw_width: np.ndarray) -> float:
    if isinstance(value, str):
        if value.startswith("auto_q"):
            q = float(value.removeprefix("auto_q")) / 100.0
            return float(np.quantile(raw_width, q))
        raise ValueError(f"Unsupported automatic gripper value: {value}")
    return float(value)


def convert_actions(raw_actions: np.ndarray, cfg: dict[str, Any]) -> ConversionResult:
    action_cfg = cfg.get("action_conversion", cfg)
    raw_layout = action_cfg.get("raw_layout", {})
    arm_idx = list(raw_layout.get("arm", range(2, 9)))
    gripper_idx = list(raw_layout.get("gripper", [0, 1]))
    gripper_cfg = action_cfg.get("gripper", {})

    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.ndim != 2:
        raise ValueError(f"Expected action matrix [N, D], got {raw_actions.shape}")
    expected_input_dim = int(action_cfg.get("input_dim", raw_actions.shape[-1]))
    expected_output_dim = int(action_cfg.get("output_dim", 8))
    if raw_actions.shape[-1] != expected_input_dim:
        raise ValueError(f"Expected raw action dim {expected_input_dim}, got {raw_actions.shape[-1]}")
    if len(arm_idx) != 7 or len(gripper_idx) != 2 or expected_output_dim != 8:
        raise ValueError("This Pi0.5 converter expects 7 arm dims, 2 gripper dims, and 8 output dims")

    arm = raw_actions[:, arm_idx]
    g0 = raw_actions[:, gripper_idx[0]]
    g1 = raw_actions[:, gripper_idx[1]]
    mode = gripper_cfg.get("mode", "same_direction")
    if mode == "same_direction":
        width = 0.5 * (g0 + g1)
    elif mode == "opposite_direction":
        width = 0.5 * (g0 - g1)
    else:
        raise ValueError(f"Unsupported gripper mode: {mode}")

    open_width = _resolve_quantile(gripper_cfg.get("open_width", "auto_q99"), width)
    closed_width = _resolve_quantile(gripper_cfg.get("closed_width", "auto_q01"), width)
    denom = open_width - closed_width
    if abs(denom) < 1.0e-8:
        raise ValueError(
            "Gripper open_width and closed_width are too close; cannot infer scalar gripper action"
        )

    output_open = float(gripper_cfg.get("output_open", 0.0))
    output_closed = float(gripper_cfg.get("output_closed", 1.0))
    alpha = (open_width - width) / denom
    scalar_unclipped = output_open + alpha * (output_closed - output_open)
    clip_low, clip_high = gripper_cfg.get("clip", [0.0, 1.0])
    scalar = np.clip(scalar_unclipped, float(clip_low), float(clip_high))
    clip_rate = float(np.mean((scalar_unclipped < clip_low) | (scalar_unclipped > clip_high)))
    actions_8d = np.concatenate([arm, scalar[:, None]], axis=1).astype(np.float32)

    return ConversionResult(
        actions_8d=actions_8d,
        raw_width=width.astype(np.float32),
        scalar_gripper_unclipped=scalar_unclipped.astype(np.float32),
        scalar_gripper=scalar.astype(np.float32),
        open_width=float(open_width),
        closed_width=float(closed_width),
        clip_rate=clip_rate,
    )


def convert_dataset(
    input_dataset: str | Path,
    output_dataset: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
    symlink_videos: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    src = LeRobotV3Dataset.from_path(input_dataset)
    dst_path = Path(output_dataset)

    if dst_path.exists() or dst_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output dataset exists: {dst_path}")
        if dst_path.is_symlink() or dst_path.is_file():
            dst_path.unlink()
        else:
            shutil.rmtree(dst_path)

    ensure_dir(dst_path / "data")
    shutil.copytree(src.path / "meta", dst_path / "meta", symlinks=True)
    copy_tree_or_symlink(src.videos_dir, dst_path / "videos", symlink=symlink_videos)

    data_files = src.iter_data_files()
    if not data_files:
        raise FileNotFoundError(f"No parquet data files under {src.data_dir}")

    raw_by_file = [column_to_numpy(pq.read_table(path, columns=["action"]), "action") for path in data_files]
    raw_all = np.concatenate(raw_by_file, axis=0).astype(np.float32, copy=False)

    calibration_mask = np.ones(raw_all.shape[0], dtype=bool)
    info = src.load_info()
    transition_contract = info.get("axis_transition_contract", {})
    terminal_padding = bool(transition_contract.get("terminal_action_padding", False))
    if terminal_padding:
        episodes = src.read_episodes_table()
        if "dataset_to_index" not in episodes.column_names:
            raise KeyError("terminal_action_padding requires episode metadata column dataset_to_index")
        episode_ends = np.asarray(episodes["dataset_to_index"].to_pylist(), dtype=np.int64)
        if (
            episode_ends.ndim != 1
            or len(episode_ends) == 0
            or np.any(episode_ends <= 0)
            or np.any(np.diff(episode_ends) <= 0)
            or int(episode_ends[-1]) != len(raw_all)
        ):
            raise ValueError(
                "terminal_action_padding requires strictly increasing dataset_to_index values "
                f"ending at {len(raw_all)}, got {episode_ends.tolist()}"
            )
        calibration_mask[episode_ends - 1] = False
    if not calibration_mask.any():
        raise ValueError("Dataset has no real action rows after terminal padding is excluded")

    calibration = convert_actions(raw_all[calibration_mask], cfg)
    resolved_cfg = copy.deepcopy(cfg)
    resolved_action_cfg = resolved_cfg.setdefault("action_conversion", {}) if "action_conversion" in cfg else resolved_cfg
    resolved_gripper_cfg = resolved_action_cfg.setdefault("gripper", {})
    resolved_gripper_cfg["open_width"] = calibration.open_width
    resolved_gripper_cfg["closed_width"] = calibration.closed_width

    all_converted: list[np.ndarray] = []
    per_file_reports: list[dict[str, Any]] = []

    for data_file, raw in zip(data_files, raw_by_file, strict=True):
        rel = data_file.relative_to(src.path)
        out_file = dst_path / rel
        ensure_dir(out_file.parent)
        table = pq.read_table(data_file)
        result = convert_actions(raw, resolved_cfg)

        action_array = pa.array(result.actions_8d.tolist(), type=pa.list_(pa.float32()))
        table = table.set_column(table.column_names.index("action"), "action", action_array)
        if cfg.get("action_conversion", {}).get("preserve_raw_action", True):
            raw_array = pa.array(raw.astype(np.float32).tolist(), type=pa.list_(pa.float32()))
            table = table.append_column("action.raw_9d", raw_array)
        pq.write_table(table, out_file, compression="zstd")

        all_converted.append(result.actions_8d)
        per_file_reports.append(
            {
                "file": str(rel),
                "rows": int(raw.shape[0]),
                "open_width": result.open_width,
                "closed_width": result.closed_width,
                "clip_rate": result.clip_rate,
            }
        )

    converted_all = np.concatenate(all_converted, axis=0)
    total_result = convert_actions(raw_all[calibration_mask], resolved_cfg)

    info_path = dst_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    features = info.setdefault("features", {})
    features["action"] = {
        **features.get("action", {}),
        "dtype": "float32",
        "shape": [8],
        "names": ["canonical_action"],
    }
    if cfg.get("action_conversion", {}).get("preserve_raw_action", True):
        features["action.raw_9d"] = {
            "dtype": "float32",
            "shape": [9],
            "names": ["raw_action"],
        }
    info["axis_train"] = {
        "source_dataset": str(Path(input_dataset).resolve()),
        "conversion_config": str(Path(config_path).resolve()),
        "canonical_action_dim": 8,
        "raw_action_dim": 9,
        "gripper_open_width": total_result.open_width,
        "gripper_closed_width": total_result.closed_width,
        "terminal_action_padding_excluded_from_calibration": terminal_padding,
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    stats_path = dst_path / "meta" / "stats.json"
    stats = src.load_stats()
    stats["action"] = compute_array_stats(converted_all[calibration_mask])
    if cfg.get("action_conversion", {}).get("preserve_raw_action", True):
        stats["action.raw_9d"] = compute_array_stats(raw_all[calibration_mask])
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    action_cfg = cfg.get("action_conversion", cfg)
    report = {
        "input_dataset": str(Path(input_dataset).resolve()),
        "output_dataset": str(dst_path.resolve()),
        "input_action_dim": int(raw_all.shape[-1]),
        "output_action_dim": int(converted_all.shape[-1]),
        "raw_action_layout": action_cfg.get("raw_layout"),
        "converted_action_layout": action_cfg.get("output_layout"),
        "gripper": {
            "mode": action_cfg.get("gripper", {}).get("mode", "same_direction"),
            "open_width": total_result.open_width,
            "closed_width": total_result.closed_width,
            "clip_rate": total_result.clip_rate,
            "dynamic_range": float(total_result.open_width - total_result.closed_width),
        },
        "checks": {
            "raw_action_finite": bool(np.isfinite(raw_all).all()),
            "converted_action_finite": bool(np.isfinite(converted_all).all()),
            "converted_dim_is_8": bool(converted_all.shape[-1] == 8),
            "gripper_range_valid": bool(
                np.nanmin(converted_all[:, 7]) >= -1.0e-6 and np.nanmax(converted_all[:, 7]) <= 1.0 + 1.0e-6
            ),
            "arm_dims_copied": bool(
                np.allclose(
                    converted_all[:, :7],
                    raw_all[:, list(action_cfg.get("raw_layout", {}).get("arm", range(2, 9)))],
                    atol=float(cfg.get("checks", {}).get("arm_copy_tolerance", 1.0e-6)),
                )
            ),
        },
        "files": per_file_reports,
        "num_frames": int(converted_all.shape[0]),
        "num_real_action_rows": int(calibration_mask.sum()),
        "terminal_action_padding_rows": int((~calibration_mask).sum()),
    }
    report_path = dst_path / "meta" / "axis_train_action_conversion_report.json"
    write_json(report_path, report)
    return report
