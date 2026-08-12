from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import numpy as np
from pydantic import BaseModel


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class OCRCandidate(BaseModel):
    raw_text: str
    normalized_text: str | None
    confidence: float
    transform: str
    structure_valid: bool
    check_digit_valid: bool
    corrections: list[str]


class ContainerEvent(BaseModel):
    event_id: str
    timestamp_utc: datetime
    camera_id: str
    track_id: str
    container_number: str
    check_digit_valid: bool
    confidence: float
    detector_confidence: float
    ocr_confidence: float
    supporting_frames: int
    first_seen_utc: datetime
    confirmed_at_utc: datetime
    bbox_xyxy: tuple[float, float, float, float]
    model_bundle_version: str


class Detector(Protocol):
    def detect(self, images: Sequence[np.ndarray]) -> list[list[Detection]]: ...


class Recognizer(Protocol):
    def recognize(self, crops: Sequence[np.ndarray]) -> list[list[OCRCandidate]]: ...
