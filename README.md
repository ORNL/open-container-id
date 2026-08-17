# Open Container ID

An open-source system to detect and recognize intermodal shipping-container identification numbers using local, hardware-accelerated deep learning.

## Status
This project is currently under active development. See `IMPLEMENTATION_STATUS.md` and `ROADMAP.md` for details.

## Instructions & Workflows

### 1. Repository Developer Setup

To develop or run tests locally:

```bash
git clone <repository>
cd open-container-id
# Install with dev dependencies
uv sync --extra dev
uv run pre-commit install
# Run the test suite
uv run pytest
```

### 2. Dataset Owner Workflow

To register and prepare data for training:

```bash
mkdir -p data/raw/downloads
# Place the downloaded datasets here
cp /path/to/pranw-v7-coco.zip data/raw/downloads/
cp /path/to/dasad-v1-coco.zip data/raw/downloads/

uv run container-id data register --config configs/data/sources.local.yaml
uv run container-id data extract --config configs/data/sources.local.yaml
uv run container-id data audit --config configs/data/sources.local.yaml
uv run container-id data build-canonical --config configs/data/canonical.yaml
uv run container-id data build-ocr --config configs/data/ocr.yaml
```

### 3. Detector Trainer

```bash
# Note: This has now been implemented.
uv sync --extra train
uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml
uv run container-id evaluate detector --run-dir runs/detector/<run-id>
uv run container-id export detector --run-dir runs/detector/<run-id>
```

### 4. OCR Trainer

```bash
# Note: The OCR trainer has now been implemented.
uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
uv run container-id evaluate ocr --run-dir runs/ocr/<run-id>
uv run container-id export ocr --run-dir runs/ocr/<run-id>
```

### 5. Offline End User

```bash
uv sync --extra runtime
uv run container-id infer image \
  --models /opt/container-id/models \
  --input sample.jpg \
  --output result.json
```

### 6. RTSP Operator

```bash
export CONTAINER_ID_RTSP_URL='rtsp://user:password@camera.example/stream'
uv run container-id rtsp run \
  --models /opt/container-id/models \
  --config configs/runtime/rtsp.example.yaml
```

## Contributing
See `CONTRIBUTING.md` (to be added) and check the `docs/issues` folder for current tasks that need to be implemented.

### 7. OSCAR Integration

To poll OSCAR for alarming occupancies and submit read container numbers back:

```bash
# Configure the connection (e.g. by setting env vars for the config)
uv run container-id oscar-poll
```
