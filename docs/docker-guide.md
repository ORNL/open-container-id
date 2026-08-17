# Docker Guide

This guide explains how to use the Docker runtime for offline deployment. The provided `Dockerfile` creates a secure, non-root environment suitable for production edge devices.

## Requirements
- Docker installed on the host machine.
- A built model bundle (see [Model Export Guide](model-export.md)).

## Building the Image

Build the runtime image using the provided `Dockerfile`. This Dockerfile only installs runtime dependencies to keep the image small and secure.

```bash
docker build -t open-container-id:latest .
```

## Security Posture

- **Non-root execution:** The container runs as a dedicated non-root user (`appuser`).
- **No network required:** Once built, the container functions entirely offline. It does not phone home, download weights, or send telemetry.

## Running Inference

You must map the model bundle and your input data into the container using volumes.

### Single Image Inference

```bash
docker run --rm \
  -v /opt/container-id/models:/models:ro \
  -v $(pwd)/data:/data:ro \
  open-container-id:latest \
  infer image \
  --models /models/container-id-models-0.1.0 \
  --input /data/sample.jpg \
  --output /tmp/result.json
```
*Note: Because the container runs as non-root, ensure the directories mapped as volumes have appropriate read permissions, and any output directories have write permissions for the container user (or output to stdout/stderr).*

### RTSP Inference

You can also run the RTSP pipeline inside the container. Ensure you map the configuration file.

```bash
docker run --rm -d \
  --name container-id-rtsp \
  -e CONTAINER_ID_RTSP_URL="rtsp://user:password@camera.example/stream" \
  -v /opt/container-id/models:/models:ro \
  -v $(pwd)/configs:/configs:ro \
  open-container-id:latest \
  rtsp run \
  --models /models/container-id-models-0.1.0 \
  --config /configs/runtime/rtsp.example.yaml
```

## Why Not Use Docker for Training?

Docker is explicitly NOT supported for training on macOS because Docker for Mac runs within a Linux VM that cannot currently leverage the Apple M-series GPU (MPS) for hardware acceleration. For training, you must use a native Python environment.