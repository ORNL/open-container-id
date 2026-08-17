# Troubleshooting Guide

This guide covers common issues and their fixes when using Open Container ID.

## Installation & Setup

**Issue: Dependency conflicts when running `uv sync`.**
- **Fix:** Ensure you are using a clean virtual environment. Delete `.venv` and re-run `uv sync --extra dev`.

## Data Preparation

**Issue: Audit fails due to duplicate images.**
- **Fix:** The automated audit uses pHash to cluster duplicates. If the canonical build fails, it is usually because the dataset contains identical images assigned to different splits (train/test leakage). Ensure you are relying on the build script to re-split the data, ignoring the original splits.

**Issue: OCR build script cannot find `dasad` dataset.**
- **Fix:** Ensure both the PranW and dasad ZIP files are placed exactly in `data/raw/downloads/` and you have run the extract step.

## macOS / MPS Training

**Issue: `NotImplementedError: The operator '...' is not currently implemented for the MPS device.`**
- **Fix:** You hit an operation not yet supported by Apple's MPS backend. Run with the fallback environment variable:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  This allows the operation to run on the CPU.

**Issue: Mac becomes unresponsive or training is extremely slow (M4 Max).**
- **Fix:** You are likely running out of unified memory and the system is heavily swapping to disk. Reduce the `batch_size` in your training configuration file.

## Inference & Runtime

**Issue: Model bundle verification fails.**
- **Fix:** The SHA-256 hash of the ONNX models does not match the manifest. The bundle may be corrupted or tampered with. Do not deploy. Redownload or rebuild the bundle.

**Issue: RTSP stream latency keeps increasing.**
- **Fix:** The inference hardware is slower than the camera's framerate, and the queue is unbound. Ensure you are using the RTSP runner (`container-id rtsp run`) which drops frames when it falls behind, rather than attempting to process every single frame sequentially. Check `queue_size` in your configuration.

**Issue: Inference falls back to CPU despite having a GPU (Linux/Windows).**
- **Fix:** Run `uv run container-id doctor` to check available execution providers. Ensure you have the correct ONNX Runtime package installed (`onnxruntime-gpu` instead of `onnxruntime`) and that your CUDA/cuDNN drivers are correctly configured.

**Issue: No output on RTSP stream.**
- **Fix:** Ensure your camera is reachable. By default, RTSP credentials are redacted in logs. If authentication is failing, verify the `CONTAINER_ID_RTSP_URL` environment variable is set correctly. Check network firewalls.