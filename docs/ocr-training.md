# OCR Training Guide

This guide covers training the Optical Character Recognition (OCR) text recognizer used to read the container numbers from cropped bounding boxes.

## Model Architecture

The system uses `docTR` for text recognition, specifically configuring a **CRNN (Convolutional Recurrent Neural Network) with a MobileNet V3 Small backbone**. This architecture is lightweight and efficient for edge deployment.

## Prerequisites

Before training, you must have completed the data preparation and built the OCR dataset (`data/processed/ocr-v1`). See the [Data Preparation Guide](data-prep.md). The OCR dataset consists of tight text crops and corresponding transcription labels.

## Configuration

Training configurations are located in `configs/train/`. The default configuration for the recognizer is `configs/train/ocr-crnn-mobilenet-v3-small.yaml`.

Key parameters include:
- `dataset`: Path to the OCR crop dataset.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size.
- `input_shape`: The fixed height and dynamic width expected by the model.

## Training Command

To start OCR training, run:

```bash
uv sync --extra train
uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
```

To run a quick one-epoch smoke test to verify your environment (especially useful on MPS):

```bash
uv run container-id train ocr \
  --config configs/train/ocr-crnn-mobilenet-v3-small.yaml \
  --override training.epochs=1
```

## Known Challenges: Vertical Text

Vertical and stacked text remains a difficult challenge for the OCR model. The current strategy relies on:
1. The detector predicting a bounding box that can be unstacked into horizontal text.
2. Generating synthetic data containing stacked text.
3. Explicit dataset augmentations focusing on rotations.

Further improvements (like a dedicated tall-aspect-ratio subset or a character-level detector) may be implemented in future iterations if required.

## Resuming Training

If interrupted, resume training with:

```bash
uv run container-id train ocr --resume runs/ocr/<run-id>
```

## Evaluation and Export

After training, evaluate the model and export to ONNX:

```bash
uv run container-id evaluate ocr --run-dir runs/ocr/<run-id>
uv run container-id export ocr --run-dir runs/ocr/<run-id>
```

See the [Evaluation Guide](evaluation.md) and [Model Export Guide](model-export.md) for more details.