# RTSP Deployment Guide

This guide explains how to deploy the Open Container ID system against live RTSP camera streams.

## The RTSP Operator

The RTSP operator component connects to an IP camera stream, decodes frames, runs the two-stage inference pipeline, and aggregates results.

### Temporal Consensus

Because single frames can be blurry or occluded, the system relies on multi-frame consensus. It maintains a bounded latest-frame queue. A container number is only reported as a successful event if it is consistently read and passes the check digit across multiple frames.

### Decode Separation

Video decoding is handled natively using `PyAV`. This is separate from the inference loop. If the hardware cannot keep up with the camera's framerate, the decoder drops frames to prevent latency build-up and memory bloat.

## Deployment Command

To start the RTSP operator, you need a model bundle and a configuration file.

```bash
export CONTAINER_ID_RTSP_URL='rtsp://user:password@camera.example/stream'
uv run container-id rtsp run \
  --models /opt/container-id/models \
  --config configs/runtime/rtsp.example.yaml
```

*Note: The environment variable `CONTAINER_ID_RTSP_URL` is used so that credentials are not hardcoded in configuration files.*

## Configuration Details

Review `configs/runtime/rtsp.example.yaml` for parameters. Key settings include:
- `fps`: Target frame rate for inference (e.g., process 2 frames per second).
- `consensus_frames`: Number of frames a number must be detected to trigger an event.
- `queue_size`: Maximum frames to hold in memory.

## Security & Privacy

- **Credential Redaction:** The runtime ensures that RTSP credentials (username/password) are redacted in all logs.
- **No Telemetry:** By default, no video frames or data are sent out of the local network.

## Testing with Webcams

For development convenience, the RTSP tool supports integer device indices (e.g., `0` for the default webcam).

```bash
export CONTAINER_ID_RTSP_URL='0'
uv run container-id rtsp run \
  --models dist/models/container-id-models-0.1.0 \
  --config configs/runtime/rtsp.example.yaml
```
*Treat webcam support as a development convenience, not a primary production deployment path.*