import uuid
from datetime import datetime
from typing import Any, Protocol


def compute_iou(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Intersection area
    inter_w = max(0.0, x2_i - x1_i)
    inter_h = max(0.0, y2_i - y1_i)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    # Union area
    box1_area = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    box2_area = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0.0:
        return 0.0

    return inter_area / union_area


class Track:
    def __init__(
        self, initial_box: tuple[float, float, float, float], timestamp: datetime
    ):
        self.track_id = str(uuid.uuid4())
        self.last_box = initial_box
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.missed_frames = 0
        self.detector_confidences: list[float] = []
        self.crop_quality_scores: list[float] = []
        self.ocr_candidates: list[Any] = []
        self.current_consensus: str | None = None
        self.emitted_event: bool = False

    def update(self, box: tuple[float, float, float, float], timestamp: datetime):
        self.last_box = box
        self.last_timestamp = timestamp
        self.missed_frames = 0


class TrackerInterface(Protocol):
    def update(
        self, detections: list[dict], current_timestamp: datetime
    ) -> list[Track]: ...


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.30, max_missed_frames: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: list[Track] = []

    def update(
        self, detections: list[dict], current_timestamp: datetime
    ) -> list[Track]:
        """
        Takes a list of generic detection dicts:
        [ {"bbox_xyxy": (x1,y1,x2,y2), ...}, ... ]
        """
        if not detections:
            # Increment missed frames for all active tracks
            for track in self.tracks:
                track.missed_frames += 1
            # Purge old tracks
            self.tracks = [
                t for t in self.tracks if t.missed_frames <= self.max_missed_frames
            ]
            return self.tracks

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        matches = []

        # Calculate distance matrix (1 - IoU)
        for d_idx, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_t_idx = -1

            box1 = tuple(det["bbox_xyxy"])

            for t_idx in unmatched_tracks:
                track = self.tracks[t_idx]
                iou = compute_iou(box1, track.last_box)

                if iou >= best_iou:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_t_idx != -1:
                matches.append((d_idx, best_t_idx))
                unmatched_tracks.remove(best_t_idx)
                unmatched_detections.remove(d_idx)

        # Update matched tracks
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            box = tuple(det["bbox_xyxy"])
            track = self.tracks[t_idx]
            track.update(box, current_timestamp)

            # Store metadata
            if "detector_confidence" in det:
                track.detector_confidences.append(det["detector_confidence"])
            if "crop_quality" in det:
                track.crop_quality_scores.append(det["crop_quality"])
            if "best_candidate" in det:
                track.ocr_candidates.append(
                    {"timestamp": current_timestamp, "candidate": det["best_candidate"]}
                )

        # Create new tracks
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            box = tuple(det["bbox_xyxy"])
            new_track = Track(box, current_timestamp)

            if "detector_confidence" in det:
                new_track.detector_confidences.append(det["detector_confidence"])
            if "crop_quality" in det:
                new_track.crop_quality_scores.append(det["crop_quality"])
            if "best_candidate" in det:
                new_track.ocr_candidates.append(
                    {"timestamp": current_timestamp, "candidate": det["best_candidate"]}
                )

            self.tracks.append(new_track)

        # Age unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].missed_frames += 1

        # Remove finalized tracks
        self.tracks = [
            t for t in self.tracks if t.missed_frames <= self.max_missed_frames
        ]

        return self.tracks
