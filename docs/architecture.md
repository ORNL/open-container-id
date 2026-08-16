# Architecture

The Open Container ID system uses a decoupled, two-stage machine learning architecture designed to provide robust text extraction from images and video streams.

## System Overview

1. **Object Detection (RF-DETR)**: The first stage is responsible for identifying the location of the ISO 6346 Container ID in the overall image. By decoupling detection from text recognition, the system can handle extreme camera angles, lighting conditions, and diverse container background colors more efficiently.
2. **Optical Character Recognition (docTR)**: The second stage extracts bounding boxes localized by the detector, rectifies them (handling orientation and crops), and passes them to a Convolutional Recurrent Neural Network (CRNN) or PARSeq model to transcribe the text.

## Inference Pipeline

When running inference (e.g., via the CLI or FastAPI service), the `RuntimePipeline` orchestration class is invoked. It performs the following sequence:

1. Takes an input image matrix (`cv2.Mat`).
2. Scales and pads it into a letterbox format acceptable by the ONNX models.
3. Invokes the ONNX Detector runtime.
4. Identifies bounding boxes meeting the threshold criteria.
5. Scores the crops for blurriness, resolution, and orientation using the crop quality heuristics.
6. Invokes the ONNX Recognizer runtime on valid crops.
7. Parses the transcribed text through an ISO 6346 format regex and Check Digit Validator.

## Video Tracking & Temporal Consensus

To support real-time RTSP/Video analysis, the system extends the single-frame pipeline with:
- **IoU Tracking**: Simple, low-overhead Intersection over Union matching associates detection bounding boxes across successive video frames.
- **Consensus Engine**: Accumulates valid reads for the same object track over time. Instead of relying on a single noisy frame, the engine emits an event only when a predefined number of temporally consistent frames confirm the container ID, and uses an aggressive Duplicate Suppressor to avoid re-emitting for parked containers.
