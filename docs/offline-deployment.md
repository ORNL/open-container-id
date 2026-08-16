# Offline Deployment

Open Container ID is designed for austere, offline deployment scenarios (e.g., ports, gates) where external API calls and network telemetry are prohibited.

## The Docker Runtime

The provided `deployment/Dockerfile.runtime` generates an image containing **only** ONNX Runtime, OpenCV, and FastAPI. It deliberately strips PyTorch to minimize surface area and image size.

1. Build the image:
```bash
docker build -t open-container-id-runtime -f deployment/Dockerfile.runtime .
```

2. Run offline inference:
```bash
docker run --rm --network none     -v "$(pwd)/models/bundle_v1:/opt/container-id/models:ro"     -v "$(pwd)/test_img.jpg:/test.jpg:ro"     open-container-id-runtime infer image --input /test.jpg --models /opt/container-id/models
```

Notice the `--network none` flag. The system is guaranteed not to request model downloads or phone home.

## Upgrades & Rollbacks

Because the application code and the model bundles are completely decoupled via the ONNX format, you can safely swap out `/opt/container-id/models` volumes without rebuilding the Docker container, making rollback instantaneous.
