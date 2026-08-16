# Evaluation

Evaluation in Open Container ID spans both individual model components and end-to-end inference logic.

## Detector Evaluation
Produces metrics focusing on localization:
- Mean Average Precision (mAP)
- Threshold calibration (finding the optimal confidence threshold that maximizes F1 score on the validation set).

```bash
uv run container-id evaluate detector --model runs/det/weights.pt --data path/to/val
```

## Recognizer Evaluation
Produces metrics focusing on text transcription:
- Word Error Rate (WER): Percentage of entire ISO 6346 strings transcribed incorrectly.
- Character Error Rate (CER): Fine-grained edit distance.

```bash
uv run container-id evaluate recognizer --model runs/ocr/weights.pt --data path/to/val
```

## End-to-End Evaluation
The ultimate measure of the system. This takes full raw images, runs the detector, runs the crop quality/orientation algorithms, runs the recognizer, and finally applies check digit verification.

A successful read must:
1. Locate the ID box.
2. Extract all 11 characters perfectly.
3. Pass the mathematical ISO 6346 check digit.

*Note: The CLI for full end-to-end evaluation over the test dataset is invoked via bash scripting iterating the `infer directory` outputs against ground truth.*
