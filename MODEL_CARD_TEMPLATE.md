# Model Card Template

This template should be filled out for each official release of an Open Container ID model bundle.

## Model Details

- **Model Version:** [e.g., 0.1.0]
- **Date Released:** [YYYY-MM-DD]
- **Developed by:** [Organization/Developer Name]
- **License:** [e.g., Apache-2.0, see DATASET_ATTRIBUTION.md for weights licensing]
- **Architectures:**
  - **Detector:** [e.g., RF-DETR Small]
  - **Recognizer:** [e.g., docTR CRNN MobileNet V3 Small]

## Intended Use

- **Primary Intended Uses:** Fully offline, local inference for reading ISO 6346 container numbers from images and video streams at intermodal facilities, ports, and terminals.
- **Primary Intended Users:** Logistics operators, automated gate systems, and researchers.
- **Out-of-Scope Uses:**
  - Real-time safety-critical systems where a false positive/negative could cause injury.
  - Tracking individuals or non-container vehicles.
  - Use in jurisdictions where privacy laws prohibit unconsented camera recording (if faces/license plates are incidentally captured).

## Training Data

- **Sources:**
  - [PranW Container Number Detection v7](https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7)
  - [dasad Container number v1](https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1)
- **Licenses:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Modifications:** [Detail any modifications, e.g., class remapping, deduplication, resplitting, cropping, and augmentation.]

## Training Methodology

- **Preprocessing:** [e.g., 640x640 resize with padding for detection; fixed height dynamic width for OCR.]
- **Split Methodology:** [e.g., Group-aware splitting by source video/family, pHash clustering to prevent train/test leakage.]
- **Hardware:** [e.g., Apple M4 Max using MPS backend.]

## Evaluation Metrics

*Metrics should be computed on the locked test split and ideally an actual-camera field test set.*

### Detector Metrics
- **mAP@50:** [value]
- **mAP@50-95:** [value]
- **Precision:** [value]
- **Recall:** [value]

### OCR Metrics
- **Word Accuracy:** [value]
- **Character Error Rate (CER):** [value]
- **End-to-End System Accuracy (Valid Check Digit):** [value]

### Metrics by Stratum
- **Daylight/Clear:** [metrics]
- **Night/Low-light:** [metrics]
- **Oblique Angles:** [metrics]

## Operational Thresholds

- **Detector Confidence Threshold:** [e.g., 0.5]
- **OCR Confidence Margin:** [e.g., minimum margin between top 2 character predictions]
- **Check Digit:** [Enforced/Not Enforced]

## Known Failure Modes

- **Vertical Text:** The recognizer struggles with stacked, vertical text. It relies on the detector proposing a bounding box that can be unstacked, or explicit synthetic data.
- **Valid-Check-Digit False Positives:** A random string of characters might occasionally satisfy the check-digit algorithm. Rely on confidence thresholds and temporal consensus across frames.
- **Heavy Damage:** Extremely rusty or physically damaged numbers may fail to detect or read correctly.

## Privacy & Security

- **Privacy Considerations:** The model runs 100% offline. No images are sent to the cloud. Care must be taken not to incidentally capture PII (like faces or license plates) without consent.
- **Operational Cautions:** Ensure RTSP credentials are not logged. Ensure the model bundle is verified via SHA-256 before deployment.

## Runtime Requirements

- **Software:** ONNX Runtime. Python 3.10+ (for `uv` based deployment) or Docker.
- **Hardware:** CPU is sufficient. Apple Silicon (CoreML), NVIDIA (CUDA/TensorRT) are supported.
- **Network:** None required for inference.

## Provenance & Checksums

- **Model Bundle Checksum (SHA-256):** [checksum]
- **Detector ONNX Checksum (SHA-256):** [checksum]
- **Recognizer ONNX Checksum (SHA-256):** [checksum]
- **Training Code Commit Hash:** [hash]