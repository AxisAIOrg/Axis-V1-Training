from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from axis_train.config import load_config
from axis_train.data import validate as validate_module
from axis_train.data.action_conversion import convert_actions, convert_dataset
from axis_train.data.lerobot_v3 import LeRobotV3Dataset, column_to_numpy
from axis_train.data.split import make_episode_split


def _write_dataset(root: Path, action_files: list[np.ndarray], *, with_video: bool = False) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": [9]},
        "action": {"dtype": "float32", "shape": [9]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    if with_video:
        features["observation.images.third_person"] = {
            "dtype": "video",
            "shape": [8, 8, 3],
        }

    lengths = [len(actions) for actions in action_files]
    total_frames = sum(lengths)
    info = {
        "fps": 30,
        "total_frames": total_frames,
        "total_episodes": len(lengths),
        "features": features,
        "axis_transition_contract": {
            "observation_rows": "recorded_state_boundaries",
            "action_rows": "outgoing_ctrl_with_terminal_padding",
            "terminal_action_padding": True,
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
    (root / "meta" / "stats.json").write_text("{}\n", encoding="utf-8")
    pq.write_table(pa.table({"task_index": [0], "task": ["synthetic task"]}), root / "meta" / "tasks.parquet")

    offset = 0
    for episode_index, actions in enumerate(action_files):
        actions = np.asarray(actions, dtype=np.float32)
        states = actions.copy()
        length = len(actions)
        table = pa.table(
            {
                "index": pa.array(np.arange(offset, offset + length), type=pa.int64()),
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 9)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 9)),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / 30.0, type=pa.float32()),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index), type=pa.int64()),
                "task_index": pa.array(np.zeros(length), type=pa.int64()),
            }
        )
        pq.write_table(table, root / f"data/chunk-000/file-{episode_index:03d}.parquet")
        offset += length

    episode_payload: dict[str, pa.Array] = {
        "episode_index": pa.array(np.arange(len(lengths)), type=pa.int64()),
        "length": pa.array(lengths, type=pa.int64()),
        "dataset_to_index": pa.array(np.cumsum(lengths), type=pa.int64()),
    }
    pq.write_table(pa.table(episode_payload), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if with_video:
        video = root / "videos" / "observation.images.third_person" / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"not a decodable video")


def _raw_actions(finger_widths: list[float], terminal_arm_value: float) -> np.ndarray:
    actions = np.zeros((len(finger_widths), 9), dtype=np.float32)
    actions[:, :2] = np.asarray(finger_widths, dtype=np.float32)[:, None]
    actions[:, 2:] = np.arange(len(finger_widths), dtype=np.float32)[:, None]
    actions[-1, 2:] = terminal_arm_value
    return actions


def test_convert_dataset_uses_global_calibration_and_excludes_terminal_padding(tmp_path: Path) -> None:
    source = tmp_path / "source"
    raw_files = [
        _raw_actions([0.00, 0.01, 0.04, 0.02], 100.0),
        _raw_actions([0.01, 0.02, 0.03, 0.02], 200.0),
    ]
    _write_dataset(source, raw_files)

    config_path = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "action_9d_to_8d.yaml"
    output = tmp_path / "converted"
    report = convert_dataset(source, output, config_path, symlink_videos=False)

    raw_all = np.concatenate(raw_files)
    real_mask = np.ones(len(raw_all), dtype=bool)
    real_mask[[3, 7]] = False
    cfg = load_config(config_path)
    calibration = convert_actions(raw_all[real_mask], cfg)
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["action_conversion"]["gripper"]["open_width"] = calibration.open_width
    resolved_cfg["action_conversion"]["gripper"]["closed_width"] = calibration.closed_width
    expected = convert_actions(raw_all, resolved_cfg).actions_8d
    actual = column_to_numpy(LeRobotV3Dataset.from_path(output).read_data_table(), "action")

    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
    assert report["num_real_action_rows"] == 6
    assert report["terminal_action_padding_rows"] == 2
    assert {entry["open_width"] for entry in report["files"]} == {calibration.open_width}
    stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert stats["action"]["count"] == [6]
    assert max(stats["action"]["max"][:7]) < 100.0


def test_video_decode_failure_fails_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _write_dataset(source, [_raw_actions([0.00, 0.02, 0.04], 1.0)], with_video=True)
    monkeypatch.setattr(
        validate_module,
        "_decode_video_sample",
        lambda path: {"ok": False, "path": str(path), "reason": "synthetic decode failure"},
    )

    summary = validate_module.validate_lerobot_dataset(
        source,
        expected_action_dim=9,
        output_dir=tmp_path / "report",
        decode_video=True,
    )

    assert not summary["ok"]
    assert any(check["name"] == "video_decodable:observation.images.third_person" and not check["ok"] for check in summary["checks"])


@pytest.mark.parametrize(
    "ratios",
    [(-0.1, 0.6, 0.5), (0.8, 0.1, 0.5), (float("nan"), 0.0, 0.0), (0.0, 0.5, 0.5)],
)
def test_episode_split_rejects_invalid_ratios(tmp_path: Path, ratios: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        make_episode_split(
            tmp_path / "unused",
            output_path=tmp_path / "split.json",
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
        )


def test_episode_split_balances_val_and_test_on_small_datasets(tmp_path: Path) -> None:
    source_5 = tmp_path / "source_5"
    _write_dataset(source_5, [_raw_actions([0.02], 1.0) for _ in range(5)])
    split_5 = make_episode_split(
        source_5,
        output_path=tmp_path / "split_5.json",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    assert len(split_5["train"]) == 3
    assert len(split_5["val_seen"]) == 1
    assert len(split_5["test_seen"]) == 1
    all_5 = sorted(split_5["train"] + split_5["val_seen"] + split_5["test_seen"])
    assert all_5 == list(range(5))

    source_3 = tmp_path / "source_3"
    _write_dataset(source_3, [_raw_actions([0.02], 1.0) for _ in range(3)])
    split_3 = make_episode_split(
        source_3,
        output_path=tmp_path / "split_3.json",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    assert len(split_3["train"]) == 1
    assert len(split_3["val_seen"]) == 1
    assert len(split_3["test_seen"]) == 1
    all_3 = sorted(split_3["train"] + split_3["val_seen"] + split_3["test_seen"])
    assert all_3 == list(range(3))

