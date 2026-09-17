"""A checkpoint written by opifex 0.2.7 or earlier (substrax's format 2) restores and upgrades.

The fixture is generated, not committed: ``scripts/make_format2_module_fixture.py`` writes
it with the substrax release that produced it. Run, in isolation::

    uv run --no-project --with "substrax==0.1.9" --with "flax==0.12.9" \\
        python scripts/make_format2_module_fixture.py tests/core/training/fixtures/format2
"""

from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore, upgrade_checkpoints

from opifex.core.training.config import TrainingConfig
from opifex.core.training.trainer import Trainer


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "format2" / "module"
FIXTURE_STEP = 7
GENERATE = (
    'uv run --no-project --with "substrax==0.1.9" --with "flax==0.12.9" '
    "python scripts/make_format2_module_fixture.py tests/core/training/fixtures/format2"
)

if not FIXTURE_ROOT.is_dir():
    raise RuntimeError(
        f"the format-2 module checkpoint is missing under {FIXTURE_ROOT}: {GENERATE}"
    )


class FixtureModel(nnx.Module):
    """The model the fixture was written from: two-input, one-output, one hidden layer."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(2, 4, rngs=rngs)
        self.linear2 = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.relu(self.linear1(x)))


def build_trainer(checkpoint_dir: Path) -> tuple[Trainer, FixtureModel]:
    """A trainer over a fresh model, checkpointing under ``checkpoint_dir``."""
    config = TrainingConfig(num_epochs=1, checkpoint_frequency=1)
    config.checkpoint_config.checkpoint_dir = str(checkpoint_dir)
    model = FixtureModel(rngs=nnx.Rngs(1))
    return Trainer(model, config), model


def test_load_checkpoint_reads_the_old_root():
    trainer, model = build_trainer(FIXTURE_ROOT)
    before = jnp.array(model.linear1.kernel[...])

    checkpoint = trainer.load_checkpoint(step=FIXTURE_STEP)

    assert checkpoint.step == FIXTURE_STEP
    assert checkpoint.metadata.epoch == 2
    assert checkpoint.metadata.metrics == {"loss": 0.25}
    assert checkpoint.metadata.extra["physics_metadata"] == {"pde": "burgers"}
    assert checkpoint.metadata.extra["upgraded_from"] == {"format_version": 2}
    assert not jnp.array_equal(model.linear1.kernel[...], before)


def test_upgrade_writes_a_format_3_root(tmp_path):
    steps = upgrade_checkpoints(FIXTURE_ROOT, tmp_path / "v3")

    assert steps == [FIXTURE_STEP]
    with OrbaxCheckpointStore(tmp_path / "v3") as store:
        metadata = store.read_metadata(FIXTURE_STEP)
    assert metadata.items == ("model",)
    assert metadata.metrics == {"loss": 0.25}
    trainer, _ = build_trainer(tmp_path / "v3")
    assert trainer.load_checkpoint(step=FIXTURE_STEP).metadata.epoch == 2
