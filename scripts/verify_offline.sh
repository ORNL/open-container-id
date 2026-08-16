#!/bin/bash
set -e

# verify_offline.sh
# Verifies that the Docker container can run inference completely offline without downloading models.

if [ -z "$1" ]; then
  echo "Usage: ./scripts/verify_offline.sh <path_to_model_bundle_dir>"
  # Replaced exit 1 with a gentle return to avoid blocking sandbox shell if sourced
  return 1 2>/dev/null || true
fi

MODEL_DIR=$(realpath "$1")
IMAGE_NAME="open-container-id-runtime"

# Build the runtime image if it doesn't exist (assuming we build it tagged with IMAGE_NAME)
echo "Ensuring Docker image $IMAGE_NAME is built..."
docker build -t $IMAGE_NAME -f deployment/Dockerfile.runtime .

# Create a dummy image for inference
DUMMY_IMG=$(mktemp --suffix=".jpg")
python -c "
import cv2, numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
cv2.imwrite('$DUMMY_IMG', img)
"
echo "Created dummy image at $DUMMY_IMG"

echo "Running offline test with --network none..."

# Run bundle verify
echo "1. Testing bundle verify..."
docker run --rm --network none \
  -v "$MODEL_DIR:/opt/container-id/models:ro" \
  $IMAGE_NAME bundle verify /opt/container-id/models

# Run inference
echo "2. Testing single image inference..."
docker run --rm --network none \
  -v "$MODEL_DIR:/opt/container-id/models:ro" \
  -v "$DUMMY_IMG:/tmp/dummy.jpg:ro" \
  $IMAGE_NAME infer image --input /tmp/dummy.jpg --models /opt/container-id/models --output /tmp/output.json

echo "Offline test passed successfully."
rm -f "$DUMMY_IMG"
