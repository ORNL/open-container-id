# Open Container ID

An open-source Python system for detecting and recognizing shipping-container ID numbers (ISO 6346) in logistics environments.

## Overview

This project implements a two-stage detection and Optical Character Recognition (OCR) pipeline:
1. **Detector**: Evaluates video frames or static images, extracting bounding boxes containing container IDs. Models are trained on RF-DETR architectures.
2. **Recognizer**: Reads the text characters inside the container ID crops. Models are based on the docTR project.

By decoupling these steps, the system provides high robustness on varying container types and camera angles, especially with the use of a Temporal Consensus Engine to suppress duplicate reads from RTSP streams.

## Features

- **Offline-first**: Run the core runtime without internet access (smoke tested via `--network none`).
- **Real-time Tracking**: RTSP stream decoding via PyAV, IoU multi-frame tracking, temporal consensus, and bounding box scoring.
- **REST Service**: Local optional FastAPI server for single-image or bulk OCR jobs.
- **ISO 6346 Verification**: All reads are stringently validated via standard ISO Check Digit rules.

## Quick Starts

### Inference from an existing Model Bundle

Make sure you have a `models/` directory correctly populated with an Open Container ID bundle.

```bash
# Setup
git clone https://github.com/tyronechrisharris/open-container-id.git
cd open-container-id
uv sync --extra runtime --extra api --extra rtsp

# Run a static image inference
uv run container-id infer image --input sample.jpg --models models/ --output result.json

# Serve the FastAPI REST endpoints locally
uv run container-id serve --models models/ --host 127.0.0.1 --port 8000
```

### Data Preparation

Data preparation assumes you have the PranW dataset available.
```bash
# Extract your datasets safely
uv run container-id data extract --archives sources.yaml --dest data/raw/

# Build the canonical dataset
uv run container-id data build-canonical --config config.yaml
```

### M4 Max / MPS Training

This project is built explicitly to support fast, localized training on Apple Silicon (MPS).
Documentation for tuning batch sizes and MPS troubleshooting can be found in `docs/training-macos-mps.md`.

```bash
# Verify MPS is active
uv run container-id train doctor

# Run detector training
uv run container-id train detector --config rfdetr_config.yaml

# Evaluate your models
uv run container-id evaluate detector --model runs/detector/weights.pt --data dataset.yaml
```

### RTSP and Offline Usage

- **RTSP**: Start a continuous processing loop over an RTSP source:
  `export CONTAINER_ID_RTSP_URL="rtsp://user:pass@10.0.0.1/stream"`
  `uv run container-id rtsp run --models models/ --config rtsp.yaml`
- **Offline Docker**: The default `deployment/Dockerfile.runtime` sets up a CPU-only Python Slim bookworm container running as the `containerid` non-root user. Read `docs/offline-deployment.md` for specific security configurations.

## Documentation

Full architectural and process guides are located in the `docs/` folder:
- [Architecture](docs/architecture.md)
- [Data Preparation](docs/data-preparation.md)
- [Dataset Audit](docs/dataset-audit.md)
- [Training on macOS (MPS)](docs/training-macos-mps.md)
- [Detector Training](docs/detector-training.md)
- [OCR Training](docs/ocr-training.md)
- [Evaluation](docs/evaluation.md)
- [Model Bundles](docs/model-bundle.md)
- [Offline Deployment](docs/offline-deployment.md)
- [RTSP Deployment](docs/rtsp-deployment.md)
- [API](docs/api.md)
- [Security & Privacy](docs/security-and-privacy.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Release Process](docs/release-process.md)

## Licensing

- **Code**: Apache License 2.0
- **Model Checkpoints (Exported)**: Apache License 2.0
- **Training Data**: CC BY 4.0. See `licenses/DATASET_ATTRIBUTION.md`.
