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
6. Create and push an annotated tag from the exact release commit.

   ```bash
   target_sha=$(git rev-parse HEAD)
   git tag -a vX.Y.Z -m "opifex X.Y.Z"
   git push origin main vX.Y.Z
   ```

7. In GitHub Actions, manually run `Publish to PyPI` with:

   - `target=github-release`
   - `version_tag=vX.Y.Z`

   The workflow verifies that the tag exists, creates the GitHub Release with
   generated release notes, then publishes to PyPI.

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

The publish workflow currently uses API token authentication. Two repo
secrets must be set on `avitai/opifex`:

- `PYPI_API_TOKEN`      — account-scoped token used by the `pypi` job
- `TEST_PYPI_API_TOKEN` — account-scoped token used by the `testpypi` job

For the maiden release of a new project, the token must be **account-scoped**
(project-scoped tokens cannot create new projects). Once the project exists
on PyPI, project-scoped tokens are preferred for least-privilege rotation.

To migrate to OIDC trusted publishing, register the project at
<https://pypi.org/manage/account/publishing/> (and the TestPyPI equivalent)
with:

- Owner: `avitai`
- Repository: `opifex`
- Workflow: `publish.yml`
- Environment: `pypi` (and `testpypi`)

Then remove the `password` inputs from the publish steps and add
`permissions: id-token: write` to each publish job.
