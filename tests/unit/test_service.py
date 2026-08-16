from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline() -> Generator[MagicMock, None, None]:
    with patch("container_id.service.app.RuntimePipeline") as mock:
        pipeline_instance = MagicMock()
        mock.return_value = pipeline_instance
        yield pipeline_instance

@pytest.fixture
def client(mock_pipeline: MagicMock, tmp_path: Any) -> Generator[TestClient, None, None]:
    import os
    os.environ["CONTAINER_ID_MODEL_DIR"] = str(tmp_path)
    from container_id.service.app import app
    with TestClient(app) as client:
        yield client

def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_infer_image_success(client: TestClient, mock_pipeline: MagicMock) -> None:
    mock_pipeline.process_frame.return_value = {
        "detections": [
            MagicMock(bbox=(0, 0, 10, 10), score=0.9, text="MSKU1234567", text_score=0.95),
            MagicMock(bbox=(10, 10, 20, 20), score=0.8, text="invalid", text_score=0.8)
        ]
    }

    # Create fake image bytes
    import cv2
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/infer",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["container_number"] == "MSKU1234567"
    assert data["container_score"] == 0.95
    assert len(data["detections"]) == 2

def test_infer_image_no_model() -> None:
    # If the model isn't loaded
    import os
    os.environ["CONTAINER_ID_MODEL_DIR"] = "invalid_path"

    with patch("container_id.service.app.RuntimePipeline") as mock:
        mock.side_effect = Exception("Model not found")
        from container_id.service.app import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.json() == {"status": "ok", "model_loaded": False}

            response = client.post(
                "/infer",
                files={"file": ("test.jpg", b"fake", "image/jpeg")}
            )
            assert response.status_code == 503
