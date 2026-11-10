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
3. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date.
4. Run the release checks.

   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check dist/*
   ```

5. Commit the version and changelog updates.
6. Open a pull request for the version and changelog commit, and merge it. The
   `main-protection` ruleset carries a `pull_request` rule with no bypass actors, so
   `main` cannot be pushed to directly; `rebase` is the only permitted merge method,
   which gives the commit a new sha.

7. Create and push an annotated tag from the release commit **as it landed on `main`**.
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

8. In GitHub Actions, manually run `Publish to PyPI` **from the `vX.Y.Z` tag**, with:

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
