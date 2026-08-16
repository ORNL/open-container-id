# Release Process

This document outlines the workflow for cutting a new semantic version (vX.X.X) of Open Container ID.

## 1. Quality Assurance
Ensure all checks pass on the `main` branch:
```bash
make setup-train
make format
make lint
make typecheck
make test
```

## 2. Version Bump
Update the version number in `pyproject.toml` and `src/container_id/version.py`. Ensure the `ROADMAP.md` and `IMPLEMENTATION_STATUS.md` reflect the completed milestones.

## 3. GitHub Tagging
Once merged into `main`, tag the release:
```bash
git tag v0.1.0
```
Note: Committing tags directly via cli script requires authenticated SSH/HTTPS git push access, which should be done manually by the maintainer.

## 4. Automation
Pushing the tag triggers the `.github/workflows/release.yml` GitHub Action.
This action will:
1. Build the Python Source Distribution (`.tar.gz`) and Wheel (`.whl`).
2. Draft a GitHub Release.
3. Attach the distributions.

## 5. Model Weights
Because Model Bundles are large, they are distributed separately from the source code.
1. Build the canonical `.onnx` bundle locally.
2. Verify the bundle: `uv run container-id bundle verify dist/models/bundle_v0.1.0`
3. Upload the resulting zipped bundle to the GitHub Release draft as an asset.
4. Publish the Release.
