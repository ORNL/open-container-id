# Detector Training Guide

This guide covers training the object detector (RF-DETR) used to locate container numbers in full images.

## Model Architecture

The system uses **RF-DETR Small** (Roboflow DETR). We chose this architecture over YOLO variants to ensure permissive licensing for commercial deployment without requiring a copyleft release of the surrounding application.

## Prerequisites

Before training, you must have completed the data preparation and audit processes to generate the canonical detection dataset (`data/processed/detection-v1`). See the [Data Preparation Guide](data-prep.md).

## Configuration

Training configurations are located in `configs/train/`. The default configuration for the detector is `configs/train/detector-rfdetr-small.yaml`.

Key configuration parameters include:
- `dataset`: Path to the canonical dataset.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size (tune this based on your hardware memory to avoid swapping).
- `learning_rate`: Initial learning rate.

## Training Command

To start training, use the following command:

```bash
uv sync --extra train
uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml
```

You can override configuration values directly from the CLI. For example, to run a one-epoch smoke test:

```bash
uv run container-id train detector \
  --config configs/train/detector-rfdetr-small.yaml \
  --override training.epochs=1
```

## Resuming Training

If a training run is interrupted, you can resume it by specifying the run directory:

```bash
uv run container-id train detector --resume runs/detector/<run-id>
```

## Evaluation and Export

After training completes, you can evaluate the model on the test set and export it to ONNX for offline deployment.

```bash
uv run container-id evaluate detector --run-dir runs/detector/<run-id>
uv run container-id export detector --run-dir runs/detector/<run-id>
```

See the [Evaluation Guide](evaluation.md) and [Model Export Guide](model-export.md) for more details.

## Hardware Acceleration

For Apple Silicon (e.g., M4 Max), training natively supports the MPS backend. See the [macOS / MPS Training Guide](training-macos-mps.md).