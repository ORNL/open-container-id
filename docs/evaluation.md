# Evaluation Guide

This guide details how to evaluate the performance of both the individual models and the end-to-end pipeline.

## Evaluation Philosophy

Public validation results on Roboflow datasets often overstate field performance. To mitigate this:
- We use a locked, group-aware test split (ensuring frames from the same video are not split across train and test sets).
- We rely on actual-camera field tests.
- We measure event-level metrics (multi-frame consensus) rather than single-frame guesses.

## Evaluating the Detector

After training the detector, evaluate it against the test split.

```bash
uv run container-id evaluate detector --run-dir runs/detector/<run-id>
```

This command will output:
- mAP@50 (Mean Average Precision at 50% IoU)
- mAP@50-95
- Precision and Recall

## Evaluating the Recognizer (OCR)

After training the recognizer, evaluate its performance on the OCR test crops.

```bash
uv run container-id evaluate ocr --run-dir runs/ocr/<run-id>
```

This command will output:
- Word Accuracy (exact match rate)
- Character Error Rate (CER)

## End-to-End Evaluation

The true test of the system is evaluating the pipeline end-to-end against full-frame images to see if it correctly reads valid ISO 6346 container numbers.

```bash
uv run container-id evaluate end-to-end --config configs/evaluate/end_to_end.yaml
```

This evaluation specifically tracks **System Accuracy (Valid Check Digit)**. A false positive detection that yields random characters is penalized unless the OCR model produces a string that coincidentally passes the check digit (which is rare).

## Using Results

The metrics output by these commands should be recorded in the [Model Card](../MODEL_CARD_TEMPLATE.md) before publishing a model bundle release.