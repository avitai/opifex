# Cloud training on vast.ai (SkyPilot)

Launch opifex training runs on rented vast.ai GPUs. SkyPilot handles provisioning,
SSH and file sync; `qh9_train.sky.yaml` declares the task and `launch.py` wires the
lifecycle (provision → sync code + data → train → fetch results → optional
teardown). Modelled on datarax's `benchmarks/sky` + `benchmarks/automation`.

## One-time setup

```bash
uv tool install "skypilot[vast]"   # or: pipx install "skypilot[vast]"
sky check vast                      # reads ~/.config/vastai/vast_api_key
```

## Launch a QH9 training run

```bash
# Reference QHNet scale (mul128) on an H100, full QH9-Stable database:
python deploy/vast/launch.py \
    --cluster opifex-qh9 \
    --env QH9_BATCH=16 --env QH9_EPOCHS=30 \
    --download-dir runs/qh9_mul128 \
    --teardown
```

`launch.py` runs `sky launch --infra vast`, which provisions the instance from
`qh9_train.sky.yaml`, rsyncs the working tree (only `src/`, `scripts/`, `deploy/`
and packaging — see `.skyignore`) and the 29 GB `QH9Stable.db`, runs
`uv sync --extra gpu --extra neural-dft`, then `scripts/train_qh9_blocks.py`. On
completion the remote run directory (`train.log`, `metrics.json`, `checkpoints/`)
is fetched into `--download-dir`.

## Knobs

- Hyper-parameters are `QH9_*` task envs (see `qh9_train.sky.yaml`); override any
  with repeated `--env KEY=VALUE`.
- `--detach` launches in the background (`sky launch -d`); stream later with
  `sky logs <cluster>` and fetch with `sky rsync <cluster>:<remote_out>/ <dir>/`.
- `--teardown` runs `sky down` after fetching; omit it to keep the instance for
  inspection (it keeps billing — `sky down <cluster>` when done).
- Instance type / disk live in `resources:` in the YAML (default `H100:1`,
  256 GB disk).

## Cost

Reference-scale runs converge fast (the architecture is data-efficient), so a full
run is typically a few hours. Check live prices with
`sky show-gpus --infra vast` before launching.
