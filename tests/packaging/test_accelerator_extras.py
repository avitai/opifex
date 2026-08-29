"""Repository contracts for the accelerator extras.

These encode decisions that were previously carried only by convention, and that drifted:
the extra was named ``gpu`` while ``setup.sh`` already spoke of a ``cuda12`` backend, it
resolved ``jax[cuda12_local]`` while the same script promised not to rely on a system CUDA
toolkit, and it carried an exact ``jax``/``jaxlib`` pin that governed the single universal
lockfile for every other extra.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

#: The accelerator extra is named for the CUDA major version, as JAX names its own extras
#: (``cuda12``, ``cuda12-local``, ``cuda13``). ``gpu`` cannot: this package also ships a
#: ``metal`` extra, and Metal is a GPU.
CUDA_EXTRA = "cuda12"


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _extras() -> dict[str, list[str]]:
    return _pyproject()["project"]["optional-dependencies"]


def _requirement_names(specifiers: list[str]) -> set[str]:
    """Return the distribution name of each specifier, without extras or version."""
    return {re.split(r"[<>=!\[; ]", specifier.strip())[0].lower() for specifier in specifiers}


def test_the_cuda_extra_is_named_for_the_cuda_version() -> None:
    """The accelerator extra is ``cuda12``, and no ``gpu`` extra remains."""
    extras = _extras()

    assert CUDA_EXTRA in extras
    assert "gpu" not in extras, "an extra named `gpu` cannot be distinguished from `metal`"


def test_the_cuda_extra_uses_the_pip_managed_variant() -> None:
    """``jax[cuda12]`` ships the CUDA wheels; ``jax[cuda12_local]`` expects a system toolkit.

    ``setup.sh`` promises the setup "does not rely on a system CUDA toolkit", so the local
    variant contradicts it, and the hand-listed NVIDIA wheels exist only because the local
    variant does not bring them.
    """
    jax_specifiers = [
        specifier for specifier in _extras()[CUDA_EXTRA] if specifier.lower().startswith("jax")
    ]

    assert jax_specifiers, "the cuda extra must declare jax"
    assert any("[cuda12]" in specifier for specifier in jax_specifiers)
    for specifier in jax_specifiers:
        assert "cuda12_local" not in specifier
        assert "cuda12-local" not in specifier


def test_no_extra_pins_jax_or_jaxlib_exactly() -> None:
    """An exact pin in any extra governs the whole lockfile, not just that extra.

    uv resolves one universal resolution across every extra, so ``jax==X`` inside an
    accelerator extra decided the jax that ``dev``, ``test`` and ``docs`` received while no
    workflow installed the accelerator extra at all.
    """
    offenders = [
        f"{extra}: {specifier}"
        for extra, specifiers in _extras().items()
        for specifier in specifiers
        if re.match(r"^jax(lib)?\b", specifier.strip(), flags=re.IGNORECASE) and "==" in specifier
    ]

    assert not offenders, f"exact jax pins govern every extra: {offenders}"


def test_the_cuda_extra_does_not_restate_the_jaxlib_lockstep() -> None:
    """Every ``jax[cuda12*]`` extra requires ``jaxlib<=X,>=X``, so a manual pin is redundant."""
    assert "jaxlib" not in _requirement_names(_extras()[CUDA_EXTRA])


def test_aggregate_extras_reference_the_cuda_extra_by_its_name() -> None:
    """``all``, ``cuda-dev`` and ``opifex-dev`` must follow the rename, or they silently drop it."""
    extras = _extras()

    for aggregate in ("all", "cuda-dev", "opifex-dev"):
        referenced = " ".join(extras[aggregate])
        assert CUDA_EXTRA in referenced, f"{aggregate} no longer includes {CUDA_EXTRA}"
        assert not re.search(r"[,\[]gpu[,\]]", referenced), f"{aggregate} still references gpu"


def test_setup_sh_syncs_the_extra_the_manifest_declares() -> None:
    """The cross-file contract that actually broke: ``--backend cuda12`` synced ``--extra gpu``."""
    setup_sh = (REPO_ROOT / "setup.sh").read_text()

    assert f"--extra {CUDA_EXTRA}" in setup_sh
    assert "--extra gpu" not in setup_sh

    # Read only the lines that build the sync command. Scanning the whole file also matches
    # prose and error strings — `die "--extra requires a value"` parses as an extra named
    # "requires" — and `--extra "$extra"` is a runtime passthrough with no literal to check.
    synced = {
        extra
        for line in setup_sh.splitlines()
        if "SYNC_ARGS" in line
        for extra in re.findall(r"--extra ([a-z0-9][a-z0-9-]*)", line)
    }

    assert {"dev", "test", CUDA_EXTRA, "metal"} <= synced, (
        f"the extractor found {synced}, so it is not reading the sync command"
    )
    for extra in synced:
        assert extra in _extras(), f"setup.sh syncs --extra {extra}, which the manifest lacks"
