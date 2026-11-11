# Releasing Opifex

Opifex publishes through `.github/workflows/publish.yml`.
No commit or tag push creates a release by itself. Release timing and versioning
stay under operator control. The manual `target=github-release` workflow path
creates a GitHub Release for an explicit existing tag with
`softprops/action-gh-release@v2` and `generate_release_notes: true`, then
publishes to PyPI with trusted publishing. Publishing an existing GitHub Release
also runs the PyPI upload path.

## Release Checklist

1. Activate the local environment.

   ```bash
   source activate.sh
   ```

2. Bump the package version in `pyproject.toml` (the static `version = "X.Y.Z"`
   field under `[project]`).
3. Check the release against the previous one. The README declares this API unstable
   and tells consumers to pin, so a break is not by itself a reason to hold a release.
   What these two checks produce is the list of downstream call sites the release moves,
   while there is still time to say so in the changelog.

   ```bash
   previous=vX.Y.Z   # Name the tag explicitly. griffe's default baseline is the latest
                     # tag reachable from HEAD, which is the wrong one on any branch
                     # cut before the last release.
   uvx --from griffe griffe check opifex -s src -a "$previous" -f oneline
   ```

   Over v0.2.10 to v0.2.11 that reports 54 breaking changes, the three documented module
   moves among them. Then run the downstream import check described below, which answers
   what `griffe` cannot: which of those breakages anyone is importing.
4. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date. Whatever step 3 reported belongs here: what moved under `Changed` with
   its new import path, what was withdrawn under `Removed`.
5. Run the release checks.

   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check dist/*
   ```

6. Commit the version and changelog updates.
7. Open a pull request for the version and changelog commit, and merge it. The
   `main-protection` ruleset carries a `pull_request` rule with no bypass actors, so
   `main` cannot be pushed to directly; `rebase` is the only permitted merge method,
   which gives the commit a new sha.

8. Create and push an annotated tag from the release commit **as it landed on `main`**.
   Read the sha after the merge rather than before it: a rebase merge renumbers the
   commit, so a tag cut from the pre-merge sha points at a commit that is not on `main`.

   ```bash
   git fetch origin main
   target_sha=$(git rev-parse origin/main)
   git tag -a vX.Y.Z -m "opifex X.Y.Z" "$target_sha"
   git push origin vX.Y.Z
   ```

   Only the tag is pushed. The ruleset's condition is `~DEFAULT_BRANCH`, and a tag is
   not a branch ref, so pushing it is unaffected.

9. In GitHub Actions, manually run `Publish to PyPI` **from the `vX.Y.Z` tag**, with:

   - `target=github-release`
   - `version_tag=vX.Y.Z`

   ```bash
   gh workflow run publish.yml --ref vX.Y.Z -f target=github-release -f version_tag=vX.Y.Z
   ```

   Run it from the tag, not from `main`. The `build` job checks out the ref the workflow
   was dispatched on without naming one, so the version in the artifact is whatever
   `pyproject.toml` says *at that ref* — the tag pins it, a branch does not.

   The workflow verifies that the tag exists, creates the GitHub Release with
   generated release notes, then publishes to PyPI. `target=github-release` does all
   three in one run, and the PyPI upload has no `skip-existing`, so a version that has
   already been published cannot be re-run. Check <https://pypi.org/project/opifex/>
   before dispatching.

## Downstream Import Check

Resolve every import the sibling checkouts make against the tree being released. Every
repository beside this one is examined; those whose manifest declares an `opifex`
dependency have their `opifex` imports collected by parsing their source, and each
statement is then executed. Run it from inside this checkout, or point
`OPIFEX_SIBLINGS` at the directory holding the sibling repositories.

```bash
uv run python - "${OPIFEX_SIBLINGS:-$(dirname "$PWD")}" <<'PY'
import ast
import pathlib
import re
import subprocess
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
unresolved = 0
for manifest in sorted(root.glob("*/pyproject.toml")):
    repo = manifest.parent
    if not (repo / ".git").exists():
        continue
    project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
    if project.get("name") == "opifex":
        continue
    required = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        required.extend(group)
    if not any(re.match(r"opifex\b", spec.strip()) for spec in required):
        continue
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    statements = set()
    for name in filter(None, listed):
        try:
            tree = ast.parse((repo / name).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if (node.module or "").split(".")[0] == "opifex":
                    bound = ", ".join(alias.name for alias in node.names)
                    statements.add(f"from {node.module} import {bound}")
            elif isinstance(node, ast.Import):
                statements.update(
                    f"import {alias.name}"
                    for alias in node.names
                    if alias.name.split(".")[0] == "opifex"
                )
    print(f"swept {repo.name}: {len(statements)} statements")
    for statement in sorted(statements):
        try:
            exec(statement, {})
        except Exception as error:
            unresolved += 1
            print(f"{repo.name}: {statement}\n    {type(error).__name__}: {error}")
print(f"unresolved: {unresolved}")
sys.exit(1 if unresolved else 0)
PY
```

A repository pinned to an exact older version cannot be reached by this release, so an
unresolved import there is not caused by it. One on an open floor (`>=`) resolves to the
new version, and an unresolved import there is a break the release ships. Stale
checkouts of a repository that has since been renamed report breaks of their own, so
read the list against what is still maintained.

Each choice in that command is load-bearing, and each was measured. Parse the source
rather than searching it line by line, because some of these import statements span
several lines inside parentheses and a line-based search truncates them without saying
so. Execute the statements rather than matching module paths, because searching the
dependents for the module paths 0.2.11 moved returns only prose — documentation,
planning notes and recorded file hashes — whereas executing their imports against the
0.2.11 tree finds a break that nothing else did:
`seismifex/src/seismifex/propagators/wave_acoustic.py` imports `SciMLSolver` and
`Solution` from `opifex.core.solver.interface`. `SciMLSolver` is defined there at
v0.2.10 and absent from v0.2.11 onward, and seismifex declares `opifex>=0.2.9`, so it
resolves to the release that removed it.

The repositories swept are printed whether or not they fail, because a sweep that
matched nothing and a release that breaks nothing both end in `unresolved: 0`. Reading
the swept list is what separates the two. Reject a run whose list omits a repository
known to consume this package.

## Manual Release Recovery

If the manual generated-release workflow is interrupted before creating the
GitHub Release, use GitHub generated notes manually from the exact tagged
commit.

   ```bash
   gh release create vX.Y.Z --target "$target_sha" --generate-notes
   ```

Publishing that release triggers the same PyPI upload workflow.

## TestPyPI

Use the manual `workflow_dispatch` path in `publish.yml` with
`target=testpypi` when validating trusted publishing setup before a real
release.

## PyPI Authentication

The publish workflow uses PyPI trusted publishing (OIDC): the `publish-pypi` and
`publish-testpypi` jobs request an `id-token` and no API token is stored in the
repository. PyPI must trust, for the project `opifex`:

- Owner: `avitai`
- Repository: `opifex`
- Workflow: `publish.yml`
- Environment: `pypi` (and `testpypi` on TestPyPI)

If PyPI rejects the upload with `invalid-publisher`, verify that registration
(<https://pypi.org/manage/account/publishing/>) before looking for secrets; the
expected identity is `repo:avitai/opifex:environment:pypi`.
