import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from container_id.runtime.pipeline import RuntimePipeline

logger = logging.getLogger(__name__)

# Global pipeline instance initialized on startup
_pipeline: RuntimePipeline | None = None

from collections.abc import AsyncGenerator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _pipeline
    bundle_dir = os.environ.get("CONTAINER_ID_MODEL_DIR", "bundle")

    try:
        logger.info(f"Loading RuntimePipeline from {bundle_dir}...")
        _pipeline = RuntimePipeline(bundle_dir)
        logger.info("RuntimePipeline loaded successfully.")
    except Exception as e: # noqa: BLE001
        logger.error(f"Failed to load RuntimePipeline: {e}")
        # Not raising an exception immediately so the app can start (perhaps for healthchecks),
        # but the /infer endpoint will return 503 if _pipeline is None.

    yield

    logger.info("Shutting down service...")
    _pipeline = None


app = FastAPI(
    title="Open Container ID Service",
    description="Local FastAPI service for container ID extraction.",
    version="0.1.0",
    lifespan=lifespan,
)


from typing import Any


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/infer")
async def infer_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run the full inference pipeline on an uploaded image.
    Returns the detections and best container ID found.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        contents = await file.read()
    except Exception as e: # noqa: BLE001 # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        import cv2
        import numpy as np

        # Decode image using cv2
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        result = _pipeline.process_frame(img)

        # Convert non-serializable types if necessary
        # process_frame returns: {"detections": [...list of Detection...]}
        # We should format it to plain dicts for FastAPI JSONResponse

        formatted_detections = []
        best_container_number = None
        best_score = -1.0

        for det in result.get("detections", []):
            formatted_detections.append({
                "bbox": det.bbox,
                "score": det.score,
                "text": det.text,
                "text_score": det.text_score
            })

            # Find the detection with the best valid container ID
            if det.text and det.text_score is not None and det.text_score > best_score:
                    # In this simple logic, we just take the highest text score.
                    # Real consensus tracking is for video, but for single image we just pick the best.
                    best_container_number = det.text
                    best_score = det.text_score

        return JSONResponse(content={
            "container_number": best_container_number,
            "container_score": best_score if best_container_number else None,
            "detections": formatted_detections
        })

    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}"
        )
