# AXIS LeRobot Format

The AXIS OpenPI patch expects a local root containing task-level LeRobot-v3
datasets. Each task directory must preserve the renderer used to reconstruct
observations and should be named:

```text
task_<id>_<observation_renderer>_state_train
```

## Directory Layout

```text
$AXIS_CLEAN_ROOT/
  task_0001_mujoco_state_train/
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.third_person/chunk-000/file-000.mp4
    videos/observation.images.wrist/chunk-000/file-000.mp4
```

## Required Columns

Frame parquet files under `data/` must contain:

- `observation.state`
- `action`
- `timestamp`
- `frame_index`
- `episode_index`
- `task_index`

Episode parquet files under `meta/episodes/` must contain:

- `dataset_to_index`
- `videos/observation.images.third_person/chunk_index`
- `videos/observation.images.third_person/file_index`
- `videos/observation.images.third_person/from_timestamp`
- `videos/observation.images.wrist/chunk_index`
- `videos/observation.images.wrist/file_index`
- `videos/observation.images.wrist/from_timestamp`

## Transition And Padding Contract

For `N` recorded state boundaries, rows `0..N-2` contain real outgoing
actuator controls and row `N-1` contains explicit terminal action padding.
Record this in `meta/info.json`:

```json
{
  "axis_transition_contract": {
    "observation_rows": "recorded_state_boundaries",
    "action_rows": "outgoing_ctrl_with_terminal_padding",
    "terminal_action_padding": true
  }
}
```

The OpenPI loader excludes every sample base whose complete action horizon
would touch that padding. For a 40-row episode, downsample ratio 2, and action
horizon 15, the valid bases are `0, 2, 4, 6, 8, 10`; terminal actions are never
clipped or repeated to fill a chunk.

The action horizon is frozen in `meta/action_transform_spec.json` and exported
as `AXIS_CLEAN_DROID_ACTION_HORIZON`. A horizon-1 compatibility view is the
strict diagnostic when deployment replans every control and a late release
must occur as a supervised first action. `axis-train finetune` resolves this
contract once before norm stats and training; `axis-train serve` reloads the
same spec from checkpoint assets.

## Numeric Layout

Raw AXIS state/action are 9D:

```text
[left_gripper_qpos, right_gripper_qpos, arm_joint_1, ..., arm_joint_7]
```

The OpenPI AXIS loader converts to DROID-style 8D:

```text
[arm_joint_1, ..., arm_joint_7, gripper_closedness]
```

Arm actions are computed as velocity-style deltas:

```text
(target_arm_qpos - current_arm_qpos) / AXIS_CLEAN_DROID_VELOCITY_SCALE
```

The scale is frozen in `meta/action_transform_spec.json` and must equal the
executed control period. For example, it is `0.2` at 5 Hz and `1 / 15` at
15 Hz. Gripper closedness maps finger width to:

```text
0 = open
1 = closed
```

At live inference, invert one returned action against the current measured
qpos, not against the first state for the whole chunk:

```text
target_arm_qpos = current_arm_qpos + action[0:7] * velocity_scale
target_finger_qpos = open_q + clip(action[7], 0, 1) * (close_q - open_q)
```

Do not clip arm velocity labels or predictions inside the transform. The
normalization statistics describe their observed range, and the simulator's
actuator limits remain the final physical safety bound. Arm clipping makes the
absolute-target round trip lossy; the recorded expert must still pass the real
Checker after forward and inverse conversion.

`axis_train.policy.AxisDroidPolicyAdapter` implements this operation and a
sequential chunk helper. The nine output positions are still logical joint
targets. A simulator adapter must read the ordered joint names from
`meta/axis_provenance.json` and resolve each joint/actuator address from the
compiled MuJoCo model; it must not assume qpos and ctrl addresses are equal.

## Useful Environment Variables

The unified `axis-train finetune` command accepts explicit path/config options
and is preferred for new runs. These variables remain supported by the lower
level `norm-stats` and `train` commands and by compatibility shims:

- `AXIS_CLEAN_ROOT`: root containing `task_*_<simulator>_state_train` directories.
- `OPENPI_AXIS_CLEAN_TASK_IDS`: comma-separated task ids to include.
- `OPENPI_AXIS_CLEAN_MAX_TASKS`: cap number of task directories for smoke tests.
- `OPENPI_AXIS_CLEAN_MAX_SAMPLES_PER_TASK`: cap sampled frames per task.
- `OPENPI_AXIS_CLEAN_VIDEO_CONTAINER_CACHE_SIZE`: PyAV container cache size.
- `AXIS_CLEAN_DROID_VELOCITY_SCALE`: arm target delta scale derived from the selected spec.
- `AXIS_CLEAN_DROID_DOWNSAMPLE_RATIO`: frame downsample ratio, default `2`.
- `AXIS_CLEAN_DROID_ACTION_HORIZON`: model horizon derived from the selected
  spec; it must not be overridden independently at training or serving time.
- `AXIS_CLEAN_DROID_CLIP_VELOCITY_ACTIONS`: spec-derived compatibility flag;
  new lossless specs set it to `false`.
- `AXIS_PI05_ASSETS_BASE_DIR`: shared absolute root where norm-stat generation
  writes and training reads `CONFIG_NAME/local/axis_clean/norm_stats.json`.
- `AXIS_PI05_LORA_INIT_PARAMS`, `AXIS_PI05_FULL_INIT_PARAMS`,
  `AXIS_PI05_ACTION_EXPERT_INIT_PARAMS`: initial weights for
  `pi05_axis_clean_local_droid_lora`, `_full` and `_action_expert`.
  `axis-train finetune --init-params` sets the one for the selected config.
  Unset, the configs fall back to `gs://openpi-assets/checkpoints/pi05_droid/params`
  (`_action_expert` tries `AXIS_PI05_FULL_INIT_PARAMS` first), so the run needs
  GCS access. Without `--init-params`, `finetune` uses
  `/data/openpi/openpi-assets/checkpoints/pi05_droid/params` when that
  directory exists.

## Validation

```bash
python data/validate_lerobot.py \
  --dataset-path "$AXIS_CLEAN_ROOT/task_0001_mujoco_state_train" \
  --expected-action-dim 9 \
  --output reports/task_0001_raw_9d \
  --decode-video
```

The validator checks metadata, required columns, episode/frame consistency,
finite numeric arrays, gripper dynamic range, and optional video decodability.
