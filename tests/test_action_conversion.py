from __future__ import annotations

import pytest
import numpy as np
from axis_train.data.action_conversion import convert_actions


def _base_cfg(**gripper_overrides):
    gripper = {
        "mode": "same_direction",
        "open_width": 0.04,
        "closed_width": 0.0,
        "output_open": 0.0,
        "output_closed": 1.0,
        "clip": [0.0, 1.0],
    }
    gripper.update(gripper_overrides)
    return {
        "action_conversion": {
            "input_dim": 9,
            "output_dim": 8,
            "raw_layout": {"arm": [2, 3, 4, 5, 6, 7, 8], "gripper": [0, 1]},
            "gripper": gripper,
        }
    }


def test_convert_actions_uses_configured_layout() -> None:
    raw = np.asarray(
        [
            [0.04, 0.04, 1, 2, 3, 4, 5, 6, 7],
            [0.00, 0.00, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )
    out = convert_actions(raw, _base_cfg()).actions_8d
    np.testing.assert_allclose(out[:, :7], raw[:, 2:9])
    np.testing.assert_allclose(out[:, 7], [0.0, 1.0])


def test_convert_actions_opposite_direction_gripper() -> None:
    raw = np.asarray(
        [
            [0.04, -0.04, 1, 2, 3, 4, 5, 6, 7],
            [0.00, 0.00, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )
    out = convert_actions(raw, _base_cfg(mode="opposite_direction")).actions_8d
    np.testing.assert_allclose(out[:, :7], raw[:, 2:9])
    np.testing.assert_allclose(out[:, 7], [0.0, 1.0])


def test_convert_actions_rejects_wrong_dim() -> None:
    raw = np.zeros((2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected raw action dim 9"):
        convert_actions(raw, _base_cfg())


def test_convert_actions_auto_quantile_gripper() -> None:
    raw = np.asarray(
        [
            [0.05, 0.05, 0, 0, 0, 0, 0, 0, 0],
            [0.01, 0.01, 0, 0, 0, 0, 0, 0, 0],
            [0.03, 0.03, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    result = convert_actions(
        raw,
        _base_cfg(open_width="auto_q99", closed_width="auto_q01"),
    )
    assert result.actions_8d.shape == (3, 8)
    assert np.all((result.actions_8d[:, 7] >= 0.0) & (result.actions_8d[:, 7] <= 1.0))
    assert result.open_width > result.closed_width
