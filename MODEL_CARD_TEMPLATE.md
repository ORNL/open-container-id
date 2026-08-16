# Model Card: Open Container ID v0.1.0

## Model Details
- **Name**: Open Container ID Checkpoints (Detector + Recognizer)
- **Version**: 0.1.0
- **Architectures**: RF-DETR Small (Detector), docTR CRNN/PARSeq (Recognizer).
- **Format**: ONNX
- **Release Date**: TBD

## Intended Use
- **Primary Use Case**: Identifying and transcribing standard ISO 6346 shipping container IDs from imagery in logistics environments.
- **Out-of-Scope Use Cases**: License plate recognition, general text OCR, or identification of non-standard codes.

## Training Data
- **Sources**: Open Source Container ID Dataset (PranW), CC BY 4.0.
- **Preprocessing**: Normalized dimensions, exact and perceptual deduplication, deterministic splitting.
- **Data Splitting**: Group-aware deterministic split (80% Train, 10% Valid, 10% Test).

## Evaluation Results
- **Metrics**: End-to-end Container ID accuracy, check-digit validation rate, tracking IoU stability.
- **Strata**: Evaluated on horizontal and vertical/tall text container alignments.
- **Thresholds**: Defined per model version in the embedded bundle `manifest.json`.

## Limitations & Known Failure Modes
- **Obscured or Dirty Text**: High likelihood of missing characters or failing check-digit validation.
- **Non-Standard Fonts**: May struggle with unusual painted geometries.
- **Vertical Orientation**: Specific failure modes expected on extreme vertical crops if not rotated properly.

## Privacy & Ethical Considerations
- These models do not process PII. They are restricted to container-level ID codes.
- Do not deploy in environments capturing human faces or vehicle license plates unless subsequent privacy-preserving redaction is in place.

## Provenance
- Hash verification is enforced via `container_id bundle verify`. See `manifest.json` for SHA256 checksums.
