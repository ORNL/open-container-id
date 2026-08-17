# macOS / MPS Training Guide

This guide describes how to train models using Apple Silicon hardware acceleration via the PyTorch MPS (Metal Performance Shaders) backend.

## Why Not Docker?
We do not use Docker for MPS training because Docker for Mac runs inside a Linux virtual machine, which currently lacks reliable pass-through support for the Apple GPU. To use hardware acceleration on a Mac, you must run Python natively on macOS.

## Prerequisites

### 1. Xcode Command-Line Tools
Ensure you have the command-line tools installed:
```bash
xcode-select --install
```

### 2. Native ARM64 Python
Verify that your Python environment is running natively (not via Rosetta 2). Use `uv` for dependency management:

```bash
uv sync --extra train
```
Check that your python is running as ARM64:
```bash
python3 -c "import platform; print(platform.machine())"
# Should print: arm64
```

## Setup and Verification

### MPS Doctor Output
Run the doctor command to ensure MPS is available and correctly configured:

```bash
uv run container-id doctor
```
Ensure the output indicates that PyTorch is using the `mps` device.

## Training

### Smoke Tests
Before running a full training job, run a one-epoch smoke test to ensure there are no MPS incompatibilities.

```bash
# Detector Smoke Test
uv run container-id train detector \
  --config configs/train/detector-rfdetr-small.yaml \
  --override training.epochs=1

# OCR Smoke Test
uv run container-id train ocr \
  --config configs/train/ocr-crnn-mobilenet-v3-small.yaml \
  --override training.epochs=1
```

### Batch-Size Tuning
The M4 Max has unified memory. Adjust the batch size in your configurations to maximize GPU utilization without triggering excessive swapping (which will cause a massive drop in training speed). Monitor memory usage via Activity Monitor.

### Fallback Behavior
Some operations may not be fully supported by MPS yet. In these cases, PyTorch might silently fall back to the CPU, or fail. The repository includes an explicit fallback flag if needed (see troubleshooting below), but major operations should run natively on MPS.

### Resume Commands
If training is interrupted, you can resume it by specifying the run directory:

```bash
uv run container-id train detector --resume runs/detector/<run-id>
uv run container-id train ocr --resume runs/ocr/<run-id>
```

### Monitoring with TensorBoard
Both RF-DETR and docTR will log metrics compatible with TensorBoard.

```bash
uv run tensorboard --logdir runs/
```

## Common MPS Errors

- **`NotImplementedError: The operator '...' is not currently implemented for the MPS device.`**
  - Fix: Update to the latest tested PyTorch version. Set the fallback flag: `export PYTORCH_ENABLE_MPS_FALLBACK=1` to allow the unsupported operation to run on the CPU (expect a performance penalty).
- **Out of Memory (OOM) / Extreme Slowdown**
  - Fix: You are likely swapping to disk. Reduce the `batch_size` in the configuration file.

## References
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple Accelerated PyTorch Training](https://developer.apple.com/metal/pytorch/)
- [Detector Training Guide](detector-training.md)
- [OCR Training Guide](ocr-training.md)