# Model Bundles

To deploy the Open Container ID system without shipping raw training checkpoints (which include optimizer states and framework dependencies like PyTorch), the system compiles artifacts into a "Model Bundle".

## Bundle Architecture

A valid Model Bundle contains:
- `detector.onnx`: The exported RF-DETR model.
- `recognizer.onnx`: The exported docTR model.
- `manifest.json`: Contains checksums (SHA256), threshold configurations, and format versions.

## Building a Bundle

Once you have trained both models, export them:

```bash
uv run container-id export detector --model runs/det/best.pt --output artifacts/detector.onnx
uv run container-id export recognizer --model runs/ocr/best.pt --output artifacts/recognizer.onnx
uv run container-id export bundle --detector artifacts/detector.onnx --recognizer artifacts/recognizer.onnx --output models/bundle_v1/
```

## Verifying a Bundle

Verification ensures the ONNX files match the SHA256 checksums inside the `manifest.json`. The runtime enforces this check.

```bash
uv run container-id bundle verify models/bundle_v1/
```

This prevents deploying tampered or corrupted models to edge networks.
