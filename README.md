# Open Container ID

An open-source system to detect and recognize intermodal shipping-container identification numbers using local, hardware-accelerated deep learning.

## What is this?
Open Container ID is a fully offline, privacy-preserving pipeline for finding and reading the ISO 6346 identification numbers on intermodal shipping containers. It uses a **two-stage pipeline**:
1. **Detection:** A lightweight model (RF-DETR) finds the text bounding boxes of container numbers.
2. **OCR (Optical Character Recognition):** A text recognizer (docTR) reads the characters in the cropped regions, with checks for validity against the ISO 6346 check digit.

### Goals
- **Fully offline and open-source:** No API keys, cloud subscriptions, or real-time internet requirements for inference.
- **Privacy preserving:** Process footage securely on your own hardware.
- **Platform agnostic inference:** Export models to ONNX to run efficiently on Linux, macOS, and edge devices.

## Documentation Reference
Detailed documentation for setup and usage can be found in the `docs/` directory:
- [Architecture Guide](docs/architecture.md)
- [Data Preparation Guide](docs/data-prep.md)
- [Dataset Audit Guide](docs/dataset-audit.md)
- [macOS / MPS Training Guide](docs/training-macos-mps.md)
- [Detector Training](docs/detector-training.md)
- [OCR Training](docs/ocr-training.md)
- [Evaluation Guide](docs/evaluation.md)
- [Model Export Guide](docs/model-export.md)
- [Offline Deployment Guide](docs/offline-deployment.md)
- [RTSP Deployment Guide](docs/rtsp-deployment.md)
- [Docker Guide](docs/docker-guide.md)
- [Security and Privacy](docs/security-and-privacy.md)
- [API Reference](docs/api-reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## Installation Variants

The project supports several installation variants depending on your needs. We recommend using `uv` for dependency management.

### 1. Runtime Inference (Quick Start)
For end users who just want to run the model on images or video:

```bash
uv sync --extra runtime
```

#### Five-Minute Inference Quick Start
Assuming you have a downloaded model bundle (e.g. `container-id-models-0.1.0`):

```bash
uv run container-id infer image \
  --models dist/models/container-id-models-0.1.0 \
  --input sample.jpg \
  --output result.json
```

See [Offline Deployment Guide](docs/offline-deployment.md) for detailed deployment instructions.

### 2. Developer Setup
For development or running tests locally:

```bash
git clone https://github.com/your-org/open-container-id
cd open-container-id
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## Data Preparation Quick Start

To prepare the system for training, dataset owners must register and audit the data.

```bash
# 1. Place raw datasets in the downloads directory
mkdir -p data/raw/downloads
cp /path/to/pranw-v7-coco.zip data/raw/downloads/
cp /path/to/dasad-v1-coco.zip data/raw/downloads/

# 2. Extract and Audit (Requires owner review)
uv sync --extra data
uv run container-id data register --config configs/data/sources.local.yaml
uv run container-id data extract --config configs/data/sources.local.yaml
uv run container-id data audit --config configs/data/sources.local.yaml

# 3. Build canonical sets for training
uv run container-id data build-canonical --config configs/data/canonical.yaml
uv run container-id data build-ocr --config configs/data/ocr.yaml
```
See the [Data Preparation Guide](docs/data-prep.md) and [Dataset Audit Guide](docs/dataset-audit.md) for full instructions.

## Model Training

### M4 Max / MPS Training Path
We fully support native training on Apple Silicon (M-series chips like the M4 Max) using the MPS backend.

```bash
# Detector Training
uv sync --extra train
uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml

# OCR Training
uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
```
See the [macOS / MPS Training Guide](docs/training-macos-mps.md), [Detector Training Guide](docs/detector-training.md), and [OCR Training Guide](docs/ocr-training.md).

## RTSP Operator Example

You can deploy the system against live RTSP camera streams. **Credentials are redacted in logs by default.**

```bash
export CONTAINER_ID_RTSP_URL='rtsp://user:password@camera.example/stream'
uv run container-id rtsp run \
  --models /opt/container-id/models \
  --config configs/runtime/rtsp.example.yaml
```
See the [RTSP Deployment Guide](docs/rtsp-deployment.md) for details.

### OSCAR Integration (Coming Soon)
Support for polling OSCAR for alarming occupancies and submitting read container numbers back is currently being implemented. See [API Reference](docs/api-reference.md).

```bash
uv run container-id oscar-poll
```

## Licensing & Privacy Defaults
- **Code:** Apache-2.0
- **Trained Weights / Models:** Requires specific attribution review based on core frameworks (see [Dataset Attribution](licenses/DATASET_ATTRIBUTION.md)).
- **Privacy:** By default, all processing happens locally. No network egress of private video frames or telemetry.

For more information, read the [Security and Privacy Guide](docs/security-and-privacy.md).

## Limitations
- Vertical text parsing is still a known challenge and requires careful dataset augmentation.
- The default detection class currently relies strictly on the PranW dataset structure which may contain check-digit false positives without the secondary OCR check.
- Real-time video processing requires proper queue bounding and frame dropping logic to prevent memory bloat over time.

## Status
This project is currently under active development. See `IMPLEMENTATION_STATUS.md` and `ROADMAP.md` for details.
