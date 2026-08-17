# Model Export Guide

This guide explains how to export trained PyTorch models to ONNX and build a unified model bundle for deployment.

## Why ONNX?

Exporting to ONNX (Open Neural Network Exchange) decouples the model weights from the PyTorch training codebase. This allows us to run inference using `onnxruntime`, which is lighter, faster, supports multiple hardware accelerators (CUDA, TensorRT, CoreML), and does not require arbitrary `.pth` code execution in production.

## 1. Exporting Models

After training and evaluating your models, export them to ONNX format.

### Export the Detector

```bash
uv run container-id export detector --run-dir runs/detector/<run-id>
```
*This command verifies that the ONNX model predictions match the PyTorch predictions within a defined parity tolerance. If it fails parity, the pipeline stops.*

### Export the Recognizer (OCR)

```bash
uv run container-id export ocr --run-dir runs/ocr/<run-id>
```

## 2. Building the Model Bundle

The inference runtime expects a unified "model bundle" containing both ONNX models and the runtime configuration.

```bash
uv run container-id bundle build \
  --detector runs/detector/<run-id>/exports/detector.onnx \
  --recognizer runs/ocr/<run-id>/exports/recognizer.onnx \
  --config configs/runtime/default.yaml \
  --output dist/models/container-id-models-0.1.0
```

## 3. Verifying the Bundle

The build process generates a manifest with SHA-256 hashes. You must verify the bundle before deployment.

```bash
uv run container-id bundle verify dist/models/container-id-models-0.1.0
```

## 4. Final Offline Verification

Run the verification script to ensure the models load and run without network access.

```bash
bash scripts/verify_offline.sh dist/models/container-id-models-0.1.0
```

Once built and verified, the bundle can be archived (`tar.gz`) and distributed for deployment. See the [Offline Deployment Guide](offline-deployment.md).