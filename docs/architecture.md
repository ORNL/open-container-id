# Architecture Guide

This document describes the high-level architecture of the Open Container ID system.

## Two-Stage Pipeline

The core logic relies on a hardware-accelerated, two-stage deep learning pipeline:

1. **Object Detection (RF-DETR):**
   - A lightweight transformer-based detector scans the input image to locate text bounding boxes corresponding to container identification numbers.
   - We use Roboflow DETR (RF-DETR) Small instead of YOLO to avoid complex licensing constraints for commercial deployment.

2. **Optical Character Recognition (docTR):**
   - The bounding boxes proposed by the detector are cropped from the original high-resolution image.
   - The `docTR` library (using a CRNN with a MobileNet V3 Small backbone) processes the crops to extract the characters.
   - The recognized text is then validated against the standard ISO 6346 check digit algorithm.

## Temporal Consensus (Video / RTSP)

For video files and RTSP streams, the system does not rely on a single frame. Instead, it employs **temporal consensus**:
- A bounded queue tracks predictions over multiple frames.
- A container number is only reported as a confident "event" when the same valid ISO 6346 string is detected consistently across multiple frames (configurable).
- This significantly reduces valid-check-digit false positives (where a wrong prediction coincidentally satisfies the check digit).

## Decoding (PyAV)

Video and RTSP stream decoding is handled natively using `PyAV` to maintain a robust connection and prevent memory bloat over time. Frames can be dropped if the inference pipeline falls behind the stream's frame rate.

## Deployment Interfaces

- **CLI:** A Typer-based CLI for all operations (training, data prep, inference, RTSP).
- **FastAPI:** (Optional) Exposes the inference engine as a microservice for external integrations.
- **Docker:** A fully containerized runtime executing as a non-root user.

## External Systems Integration

- **OSCAR API:** An integration allows polling OSCAR for alarming occupancies and submitting recognized container numbers back for adjudication. (See [API Reference](api-reference.md)).