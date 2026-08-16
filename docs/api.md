# FastAPI Service API

A local FastAPI service is available to allow integration with external dashboards, gates, or custom orchestration software.

## Starting the Service

Ensure you installed the `api` extra:
```bash
uv sync --extra runtime --extra api
```

```bash
uv run container-id serve --models /opt/container-id/models --host 127.0.0.1 --port 8000
```

## Endpoints

### `GET /health`
Returns the status of the API and verifies if the Model Bundle loaded correctly.
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /infer`
Upload an image via `multipart/form-data` to run the full inference pipeline.

**Request:**
```bash
curl -X POST -F "file=@truck.jpg" http://127.0.0.1:8000/infer
```

**Response:**
```json
{
  "container_number": "MSKU1234567",
  "container_score": 0.98,
  "detections": [
    {
      "bbox": [100, 200, 300, 400],
      "score": 0.95,
      "text": "MSKU1234567",
      "text_score": 0.98
    }
  ]
}
```
If no valid check-digit container ID is found, `container_number` will be `null`.
