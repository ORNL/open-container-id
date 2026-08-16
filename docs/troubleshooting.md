# Troubleshooting

## Model Loading Failures
**Error:** `Invalid protobuf file` or `Manifest verification failed`
- **Cause:** Your ONNX bundle is corrupted, or you are pointing to the wrong directory.
- **Fix:** Redownload the bundle, or rebuild it using `uv run container-id export bundle ...`. Run `uv run container-id bundle verify <path>`.

## MPS Device Errors (macOS)
**Error:** `NotImplementedError: The operator 'aten::xxx' is not currently implemented for the MPS device.`
- **Cause:** PyTorch MPS (Metal) backend lacks support for an operation used by docTR or RF-DETR.
- **Fix:** Set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running your training command.

## Dependency Errors
**Error:** `ModuleNotFoundError: No module named 'av'`
- **Cause:** You tried to run an RTSP command but didn't install the optional extra.
- **Fix:** `uv sync --extra runtime --extra rtsp`.

## No Containers Detected
- **Cause:** If evaluating against your own dataset, the containers might not match the size/aspect distribution of the training data. Or your `confidence_threshold` is too high.
- **Fix:** Adjust the threshold in the `manifest.json` or retrain the detector using your field images in the canonical build.
