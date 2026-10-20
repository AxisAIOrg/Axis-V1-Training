from __future__ import annotations

import numpy as np
from axis_train.data.action_conversion import convert_actions


def test_convert_actions_uses_configured_layout() -> None:
    raw = np.asarray(
        [
            [0.04, 0.04, 1, 2, 3, 4, 5, 6, 7],
            [0.00, 0.00, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )
    cfg = {
        "action_conversion": {
            "input_dim": 9,
            "output_dim": 8,
            "raw_layout": {"arm": [2, 3, 4, 5, 6, 7, 8], "gripper": [0, 1]},
            "gripper": {
                "mode": "same_direction",
                "open_width": 0.04,
                "closed_width": 0.0,
                "output_open": 0.0,
                "output_closed": 1.0,
                "clip": [0.0, 1.0],
            },
        }
    }
    out = convert_actions(raw, cfg).actions_8d
    np.testing.assert_allclose(out[:, :7], raw[:, 2:9])
    np.testing.assert_allclose(out[:, 7], [0.0, 1.0])


def test_convert_actions_defaults_to_canonical_axis_franka_layout() -> None:
    raw = np.asarray(
        [
            [0.04, 0.04, 10, 20, 30, 40, 50, 60, 70],
            [0.00, 0.00, 11, 21, 31, 41, 51, 61, 71],
        ],
        dtype=np.float32,
    )
    # Minimal config without specifying raw_layout
    cfg = {
        "action_conversion": {
            "gripper": {
                "open_width": 0.04,
                "closed_width": 0.0,
            }
        }
    }
    out = convert_actions(raw, cfg).actions_8d
    # Canonical 9D layout: gripper is [0, 1], arm is [2..8]
    # In 8D DROID output: arm [0..6] maps to raw [2..8], gripper 7 is scalar
    np.testing.assert_allclose(out[:, :7], raw[:, 2:9])
    np.testing.assert_allclose(out[:, 7], [0.0, 1.0])

