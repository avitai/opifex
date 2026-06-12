# Cloud training on vast.ai

Launch opifex training runs on rented vast.ai GPUs. Two paths:

- **`launch_direct.py` (recommended)** — drives the `vastai` CLI directly for
  deterministic hardware (provision a specific single-GPU offer → sync code + db →
  build the env → train in `tmux`). Use this for cards SkyPilot's catalog can't
  target (e.g. the RTX PRO 6000 server variant).
- **`launch.py` + `qh9_train.sky.yaml`** — the SkyPilot path (modelled on
  datarax's `benchmarks/sky` + `benchmarks/automation`). Convenient for H100/A100,
  but see the *SkyPilot caveat* below.

## Direct launcher (recommended)

```bash
python deploy/vast/launch_direct.py \
    --gpu RTX_PRO_6000_S --disk 256 \
    --db /mnt/ssd2/Data/qh9/raw/QH9Stable.db \
    --ssh-key ~/.ssh/<key> --fetch runs/qh9_mul128 \
    -- --hidden "128x0e + 128x1o + 128x2e + 128x3o + 128x4e" \
       --sh-lmax 4 --num-interactions 5 --start-refinement-layer 2 \
       --bottleneck-mul 32 --batch-size 16 --epochs 30
```

It registers the SSH key with vast, provisions the cheapest matching single-GPU
offer (a plain CUDA image; the onstart installs sshd and injects the key),
rsyncs `src/`/`scripts/`/`deploy/` and the database, runs
`uv sync --extra gpu --extra neural-dft` then **forces the bundled
`jax[cuda12]==0.8.0`** (so it runs on Blackwell regardless of the image's CUDA),
and starts `scripts/train_qh9_blocks.py` in a `tmux` session. Args after `--` are
forwarded to the trainer. Monitor / teardown commands are printed at the end.

## SkyPilot caveat

SkyPilot's vast support mis-provisions unless `vastai-sdk` is pinned `>=0.2.6,<0.3`
(1.0+ breaks its offer filtering, so it silently picks an arbitrary machine);
install it as `uv tool install "skypilot[vast]>=0.11,<0.12" --with
"vastai-sdk>=0.2.6,<0.3"`. Its catalog also lacks newer cards (e.g. RTX PRO 6000
server) — use the direct launcher for those.

## SkyPilot path

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
