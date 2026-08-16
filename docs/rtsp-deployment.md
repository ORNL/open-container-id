# RTSP Deployment

The system contains an optimized pipeline for decoding RTSP video streams from IP cameras, running temporal tracking, and emitting single confident Container ID reads.

## Configuration

RTSP logic requires PyAV (`av` package). Ensure you installed the `rtsp` extra:
```bash
uv sync --extra runtime --extra rtsp
```

Create a YAML configuration (`rtsp.yaml`):
```yaml
schema_version: 1
camera:
  id: "gate_1_inbound"
  url_env: "CONTAINER_ID_RTSP_URL"
  selected_frame_rate: 5.0
  reconnect:
    initial_delay_seconds: 1
    maximum_delay_seconds: 30
```

## Running the Pipeline

Inject the RTSP credentials securely via the environment variable:

```bash
export CONTAINER_ID_RTSP_URL="rtsp://user:secret@10.0.1.50:554/stream1"
uv run container-id rtsp run --models /opt/models --config rtsp.yaml
```

## Internals
- **LatestFrameQueue**: Decodes frames but drops them if the ML pipeline is backlogged. This guarantees we always process the freshest frame instead of building massive memory queues that drift from real-time.
- **IoU Tracking & Consensus**: Keeps track of moving containers. A container ID is only emitted when 3 temporally contiguous frames agree on the check-digit-validated text.
