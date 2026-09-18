r"""Write the format-2 module checkpoint the migration test restores.

opifex 0.2.7 and earlier saved a trainer checkpoint through substrax's format-2 store as
``store.save(model, step=step, loss=loss, physics_metadata=..., additional_metadata={"step":
step, "epoch": epoch})``: the module's state as the one payload beside a JSON sidecar. opifex
now writes format 3 and reads such a root through substrax's module-only layout; this script
writes one with the substrax release that produced it, so the test reads a real one. Run it
in isolation, never from the project venv::

    python3 scripts/write_format2_fixture.py tests/core/training/fixtures/format2

The fixture is a few kilobytes: a two-input, one-hidden-layer, one-output module at step 7.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore


STEP = 7


class FixtureModel(nnx.Module):
    """The model ``tests/core/training/test_checkpoint_format2.py`` rebuilds."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(2, 4, rngs=rngs)
        self.linear2 = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.relu(self.linear1(x)))


def main(argv: list[str] | None = None) -> int:
    """Write the fixture under the root named on the command line."""
    parser = argparse.ArgumentParser(
        description="Write the format-2 module checkpoint the migration test restores."
    )
    parser.add_argument("root", type=Path, help="Directory the fixture is written under")
    root = parser.parse_args(argv).root / "module"
    if root.exists():
        shutil.rmtree(root)

    model = FixtureModel(rngs=nnx.Rngs(42))
    with OrbaxCheckpointStore(root, max_to_keep=None) as store:
        path = store.save(
            model,
            step=STEP,
            loss=0.25,
            physics_metadata={"pde": "burgers"},
            additional_metadata={"step": STEP, "epoch": 2},
        )
    sys.stdout.write(f"module: step {STEP} at {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
