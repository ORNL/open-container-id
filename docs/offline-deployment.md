# Offline Deployment Guide

This guide explains how to deploy the Open Container ID system in offline, air-gapped environments without any reliance on internet connectivity for inference.

## Model Bundles

The system uses a unified "model bundle" that packages the trained ONNX detector, the ONNX OCR model, and required configuration files into a single distributable directory.

### Installation

1. **Obtain the bundle:** Transfer the built model bundle (e.g., `container-id-models-0.1.0.tar.gz`) to the target machine via secure means (e.g., USB drive).
2. **Extract the bundle:**
   ```bash
   tar -xzf container-id-models-0.1.0.tar.gz -C /opt/container-id/models
   ```

### Hash Verification

Before using the bundle, verify its integrity. The system provides a built-in verification tool that checks the SHA-256 hashes of all models against the bundle's manifest.

```bash
uv run container-id bundle verify /opt/container-id/models/container-id-models-0.1.0
```
*Note: If a hash fails, do not deploy the model.*

## Deployment Paths

### 1. Native Path (Recommended for Edge)
Run directly on the host OS for maximum performance and direct hardware access.

```bash
uv sync --extra runtime
uv run container-id infer image \
  --models /opt/container-id/models/container-id-models-0.1.0 \
  --input sample.jpg
```

### 2. Docker Path
Run securely within a container. The Docker runtime is designed to run non-root and functions completely offline. See the [Docker Guide](docker-guide.md).

```bash
docker run --rm -v /opt/container-id/models:/models -v $(pwd):/data \
  open-container-id:latest \
  infer image --models /models/container-id-models-0.1.0 --input /data/sample.jpg
```

## ONNX Runtime Providers

The runtime automatically attempts to use hardware acceleration via ONNX Runtime Execution Providers if available.

- **Linux / Windows:** Will attempt `CUDAExecutionProvider` or `TensorrtExecutionProvider` if installed; falls back to `CPUExecutionProvider`.
- **macOS:** Will attempt `CoreMLExecutionProvider`; falls back to `CPUExecutionProvider`.

You can inspect the active providers using the doctor command:
```bash
uv run container-id doctor
```

## Offline Test

To ensure the system is completely independent of the network, you should perform a network-disabled smoke test prior to final deployment.

```bash
# On Linux, use unshare to run without network namespaces
unshare -r -n uv run container-id infer image \
  --models /opt/container-id/models/container-id-models-0.1.0 \
  --input sample.jpg
```
Alternatively, physically disconnect the network cable or disable the NIC, reboot, and verify inference still starts up successfully.

## Upgrading Models

Because the application logic is decoupled from the model weights, upgrading models does not require changing the application code.

1. Extract the new bundle to a new directory (e.g., `container-id-models-0.2.0`).
2. Verify the new bundle:
   ```bash
   uv run container-id bundle verify /opt/container-id/models/container-id-models-0.2.0
   ```
3. Update the symbolic link or configuration to point to the new bundle path:
   ```bash
   ln -sfn /opt/container-id/models/container-id-models-0.2.0 /opt/container-id/models/latest
   ```
4. Restart the inference service.

## Rollback

If a new model bundle performs poorly, rollback is immediate:

1. Re-point the symlink or configuration back to the previous bundle:
   ```bash
   ln -sfn /opt/container-id/models/container-id-models-0.1.0 /opt/container-id/models/latest
   ```
2. Restart the inference service.
