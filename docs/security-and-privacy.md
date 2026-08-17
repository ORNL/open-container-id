# Security and Privacy Guide

This guide details the security practices and privacy defaults built into the Open Container ID system.

## Privacy Defaults

- **Fully Offline:** The runtime is designed to operate in air-gapped environments. It does not phone home, require API keys, download models at runtime, or send telemetry data.
- **No Private Frame Egress:** All image and video processing happens locally. Frames are not sent to any cloud service.
- **Incidental PII:** The system is intended to read container identification numbers. However, cameras may incidentally capture Personally Identifiable Information (PII) such as faces or vehicle license plates. Because processing is local and offline, this data remains under your control. Ensure your camera placement complies with local privacy laws regarding unconsented recording.

## Security Practices

### Safe Archive Extraction
The data preparation pipeline includes mitigations against malicious dataset archives (e.g., ZIP path traversal, decompression bombs). The extraction process strictly validates file paths and bounds extracted sizes.

### Image and Video Decoding
The system enforces configurable maximum upload bytes and decoded-size limits (using Pillow/OpenCV settings) to prevent out-of-memory denial-of-service attacks from malformed images.

### RTSP Credential Handling
Deploying against live cameras requires passing credentials (username/password) in the RTSP URL.
- **Configuration:** Credentials are passed via environment variables (e.g., `CONTAINER_ID_RTSP_URL`) rather than hardcoded in YAML config files.
- **Redaction:** The application logic redacts the credentials portion of the URI before writing to any application logs.

### Model Bundle Integrity
Arbitrary PyTorch `.pth` files can execute malicious code when loaded. To mitigate this:
1. The system exports to and runs ONNX models in production (`onnxruntime` is generally safer than unpickling Python objects).
2. The runtime includes a `bundle verify` command that checks SHA-256 hashes of the models against the bundle manifest before loading. Do not load bundles that fail hash verification.

### API Security (OSCAR Integration)
If the optional FastAPI service or OSCAR polling integration is used:
- By default, the API binds only to loopback (`127.0.0.1`).
- Optional token authentication can be enabled for the endpoints.

### Docker Non-Root Runtime
The provided `Dockerfile` creates an image that executes the inference engine as a non-root user, limiting the blast radius of any potential container breakout.