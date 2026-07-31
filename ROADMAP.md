# Project Roadmap

This document outlines the planned work to bring Open Container ID to version 0.1.0.
For detailed information on each task, please refer to the corresponding issue file in the `docs/issues/` directory.

## Current Milestones

### Milestone 1 & 2: Project Setup and Core Validation
- [x] Issue 1: Scaffold package, CLI, quality tooling, and CI.
- [x] Issue 2: Implement ISO 6346 parser and check digit.
- [x] Issue 3: Implement safe dataset archive registration and extraction.
- [x] Issue 4: Implement COCO validator.

### Milestone 3: Data Pipeline
- [ ] Issue 5: Implement source filename label parser.
- [ ] Issue 6: Build dataset audit and contact sheets.
- [ ] Issue 7: Add PranW class-semantic confirmation gate.
- [ ] Issue 8: Implement exact and perceptual duplicate grouping.
- [ ] Issue 9: Implement group-aware deterministic split.

### Milestone 4: Training Setup
- [ ] Issue 10: Build canonical one-class COCO dataset.
- [ ] Issue 11: Build OCR crop dataset and docTR labels.
- [ ] Issue 12: Implement manual review store and UI/CSV flow.
- [ ] Issue 13: Implement MPS device doctor and training run manifests.

### Milestone 5: Model Training & Export
- [ ] Issue 14: Implement RF-DETR Small training adapter.
- [ ] Issue 15: Implement detector metrics and threshold calibration.
- [ ] Issue 16: Implement detector ONNX export and parity tests.
- [ ] Issue 17: Implement docTR CRNN training adapter.
- [ ] Issue 18: Implement OCR metrics and error analysis.
- [ ] Issue 19: Implement recognizer ONNX export and parity tests.

### Milestone 6: Runtime and API
- [ ] Issue 20: Implement model-bundle manifest and verification.
- [ ] Issue 21: Implement ONNX detector runtime.
- [ ] Issue 22: Implement ONNX recognizer runtime.
- [ ] Issue 23: Implement crop-quality and orientation transforms.
- [ ] Issue 25: Implement still-image and directory inference CLI.

### Milestone 7: Advanced Tracking and Streams
- [ ] Issue 26: Implement IoU tracker.
- [ ] Issue 27: Implement temporal consensus and duplicate suppression.
- [ ] Issue 28: Implement video inference.
- [ ] Issue 29: Implement RTSP decoder, bounded queue, and reconnect.

### External Integrations

- [ ] Issue 41: OSCAR OCR Integration API

### Milestone 8: Deployment and Polish
- [ ] Issue 30: Implement local FastAPI service.
- [ ] Issue 31: Implement non-root runtime Docker image.
- [ ] Issue 32: Implement offline no-network smoke test.
- [ ] Issue 33: Complete licensing, attribution, model card, and citation files.
- [ ] Issue 40: Prepare v0.1.0 release.

### Deferred / Experimental
- [ ] Issue 24: Implement vertical-unstack experiment.
- [ ] Issue 34: Add field-data import path and protected field-test split.
- [ ] Issue 35: Benchmark RF-DETR Nano versus Small.
- [ ] Issue 36: Benchmark CRNN versus PARSeq.
- [ ] Issue 37: Add optional synthetic OCR generator.
- [ ] Issue 38: Add Core ML export and parity experiment.
- [ ] Issue 39: Add multi-camera scheduling design.
