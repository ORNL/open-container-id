# OCR Training

After detector training, the bounding boxes representing container numbers must be read. We employ models based on the Mindee docTR ecosystem (specifically CRNN or PARSeq).

## Preparing Crops

The OCR model expects tightly cropped image regions of the text, not full shipping yard views.

```bash
uv run container-id data build-ocr --canonical data/processed/detection-v1 --output data/processed/ocr-v1
```

## Running the Training Job

Start the CRNN training:

```bash
uv run container-id train recognizer --config crnn_config.yaml
```

The configuration dictates:
- Sequence length constraints (typically 11 chars for ISO 6346).
- Valid vocabulary (A-Z, 0-9).
- Batch sizing.

## Evaluation & Error Analysis

```bash
uv run container-id evaluate recognizer --model runs/ocr/latest/best.pt --data data/processed/ocr-v1/test
```

This logs Character Error Rate (CER) and Word Error Rate (WER). Pay special attention to the tall/vertical crop strata metrics, as vertically painted container IDs are the most common source of failure.
