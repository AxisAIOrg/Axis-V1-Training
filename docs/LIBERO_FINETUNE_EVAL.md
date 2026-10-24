# LIBERO Fine-Tuning And Evaluation

LIBERO support is provided through OpenPI's existing examples. This repository
adds release wrappers and documentation, but does not vendor LIBERO assets.

## Install LIBERO Dependencies

Initialize OpenPI submodules from the patched OpenPI checkout:

```bash
cd third_party/openpi
git submodule update --init --recursive
```

OpenPI recommends Docker for LIBERO evaluation:

```bash
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

For a non-Docker setup, create the LIBERO example environment:

```bash
cd third_party/openpi
uv venv --python 3.8 examples/libero/.venv
uv pip sync \
  --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero
```

## Convert LIBERO Data

```bash
cd "$OPENPI_ROOT"
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir /path/to/libero_rlds
```

The converted dataset is consumed by the OpenPI configs `pi05_libero_lora`
and `pi05_libero`.

## Fine-Tune Pi0.5 On LIBERO

```bash
cd "$OPENPI_ROOT"
uv run scripts/train.py pi05_libero \
  --exp-name libero_pi05_v001 \
  --overwrite
```

For lower-memory Pi0.5 LoRA fine-tuning:

```bash
uv run scripts/train.py pi05_libero_lora \
  --exp-name libero_pi05_lora_v001 \
  --overwrite
```

`pi05_libero_lora` reads its initial weights from `LIBERO_PI05_LORA_INIT_PARAMS`
and its norm-stat assets from `LIBERO_PI05_ASSETS`. Unset, they default to
`gs://openpi-assets/checkpoints/pi05_base/params` and
`gs://openpi-assets/checkpoints/pi05_libero/assets`, which need GCS access.
Point them at local copies to train offline:

```bash
LIBERO_PI05_LORA_INIT_PARAMS=/path/to/pi05_base/params \
LIBERO_PI05_ASSETS=/path/to/pi05_libero/assets \
uv run scripts/train.py pi05_libero_lora \
  --exp-name libero_pi05_lora_v001 \
  --overwrite
```

For resumed training:

```bash
uv run scripts/train.py pi05_libero_lora \
  --exp-name libero_pi05_lora_v001 \
  --resume
```

## Serve A LIBERO Policy

From this repo:

```bash
export OPENPI_ROOT="$PWD/third_party/openpi"
axis-train serve --serve-env LIBERO \
  pi05_libero \
  "$OPENPI_ROOT/checkpoints/pi05_libero/libero_pi05_v001/29999"
```

## Evaluate On LIBERO

From this repo:

```bash
axis-train libero-eval \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 50 \
  --args.video-out-path data/libero/videos
```

Common suite names include `libero_spatial`, `libero_object`, `libero_goal`,
and `libero_10`; use the names supported by the installed OpenPI/LIBERO
checkout.

## Expected Evaluation Flow

1. Start the OpenPI policy server with `axis-train serve --serve-env LIBERO`.
2. Start LIBERO evaluation with `axis-train libero-eval` (it delegates to
   `examples/libero/main.py` in the isolated LIBERO environment).
3. Confirm the evaluator is connecting to the policy server.
4. Save videos and success metrics under a run-specific output directory.
