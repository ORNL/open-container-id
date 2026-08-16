# Security & Privacy

Open Container ID is built with privacy-by-default and strict operational security in mind.

## Privacy

- **No Facial/License Plate Recognition**: The models are trained strictly on shipping container ISO 6346 text and container bounding boxes. They do not possess the vocabulary or feature-extraction capabilities to recognize human faces or passenger vehicle license plates.
- **No Persistence**: The runtime Docker image and `RTSPRunner` do not persist video frames or images to disk by default. Only extracted text events are logged to the `events.jsonl` stream.

## Security

- **Non-root Docker**: The provided `Dockerfile.runtime` executes as the `containerid` non-root user.
- **Offline Integrity**: The `bundle verify` command uses SHA256 checksums inside the `manifest.json` to cryptographically verify that the ONNX weights have not been tampered with or corrupted since export.
- **Network Isolation**: Open Container ID never phones home. It downloads no weights at runtime. The entire stack can run in a `--network none` container.
- **Credential Masking**: The `rtsp run` command parses the RTSP URI specifically to mask passwords before outputting any logs.
