# API Reference

This document provides a reference for the primary interfaces to the Open Container ID system.

## Command Line Interface (CLI)

The CLI is implemented using Typer and is the primary way to interact with the system.

### Entrypoint
All commands are accessed via the `container-id` entrypoint (when installed via `uv`).

### Data Preparation
- `container-id data register`: Register dataset sources.
- `container-id data extract`: Safely extract archives.
- `container-id data audit`: Run automated dataset audits.
- `container-id data acknowledge-audit`: Acknowledge semantic mappings.
- `container-id data review`: Manually review flagged samples.
- `container-id data build-canonical`: Build the detector dataset.
- `container-id data build-ocr`: Build the OCR dataset.
- `container-id data summarize`: Print dataset statistics.

### Training & Export
- `container-id train detector`: Train the RF-DETR model.
- `container-id train ocr`: Train the docTR OCR model.
- `container-id evaluate detector`: Evaluate the detector.
- `container-id evaluate ocr`: Evaluate the OCR model.
- `container-id evaluate end-to-end`: Evaluate the full pipeline.
- `container-id export detector`: Export detector to ONNX.
- `container-id export ocr`: Export OCR to ONNX.

### Bundling
- `container-id bundle build`: Create a distributable model bundle.
- `container-id bundle verify`: Verify bundle integrity using SHA-256 hashes.

### Inference & Runtime
- `container-id infer image`: Run inference on a single image.
- `container-id rtsp run`: Start the RTSP video stream processor.
- `container-id doctor`: Inspect the hardware and ONNX Execution Providers.

## FastAPI Service (Optional)

For integrations requiring a microservice architecture, the system provides a FastAPI wrapper.

*(Note: The FastAPI implementation is currently a stub and is being actively developed.)*

- **Host/Port:** Default binds to `127.0.0.1:8000`.
- **Endpoints:**
  - `POST /infer`: Submit an image file, receive a JSON response with bounding boxes and recognized text.
  - `GET /health`: Health check and model bundle verification status.

## OSCAR Integration API (Coming Soon)

We are actively developing an integration to poll the OSCAR backend for alarming occupancies and submit read container numbers back via the Adjudication command endpoint.

*(Note: This feature is tracked in Issue 41 and is pending implementation.)*

### Planned Functionality
- Authenticate with OSCAR.
- Retrieve alarming occupancies by polling.
- Download video files for processing.
- Process video with OCR.
- Submit extracted container numbers back to OSCAR.

```bash
uv run container-id oscar-poll
```