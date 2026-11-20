"""opifex's library code against substrax's Flax NNX source rules.

``substrax.testing.nnx_rules`` reads each file's syntax tree, without importing it, and
reports every site of a defect class measured across the stack's libraries: a seed written
into library code, a probe or fallback around ``nnx.Rngs``, a mode flag defaulting to a bool,
``trainable=`` metadata flax never reads, the salted builtin ``hash`` and a no-op
``super().__init__()`` in a direct ``nnx.Module`` subclass. The rules' docstrings give each
reason.

A site the rule cannot tell from a legitimate use is exempted on its own line by a comment
naming the rule and the reason, ``# substrax-allow: <rule>: <reason>``.
"""

from pathlib import Path

from substrax.testing.nnx_rules import findings


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = ("src/opifex",)
_RULES = ("noop-super-init", "trainable-metadata")


def test_library_code_has_no_unexempted_site() -> None:
    """Every rule holds over ``src/opifex``, exemptions aside."""
    found = findings(_REPO_ROOT, _LIBRARY, rules=_RULES)
    assert found == [], "\n".join(f"{f.path}:{f.line} [{f.rule}] {f.source}" for f in found)


def test_a_site_planted_in_a_copy_of_the_library_is_found(tmp_path: Path) -> None:
    """The scan reads opifex's own files: a site planted in a copy of one is reported."""
    module = Path("src/opifex/neural/base.py")
    original = (_REPO_ROOT / module).read_text(encoding="utf-8")
    planted = tmp_path / module
    planted.parent.mkdir(parents=True)
    planted.write_text(
        original + "\n\nclass _Planted(nnx.Module):\n    def __init__(self) -> None:\n"
        "        super().__init__()\n",
        encoding="utf-8",
    )
    planted_line = len(original.splitlines()) + 5
    found = findings(tmp_path, _LIBRARY, rules=("noop-super-init",))
    assert (module.as_posix(), planted_line) in {(f.path, f.line) for f in found}
