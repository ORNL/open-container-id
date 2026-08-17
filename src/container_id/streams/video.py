import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
from tqdm import tqdm

from container_id.runtime.consensus import ConsensusEngine
from container_id.runtime.pipeline import RuntimePipeline
from container_id.runtime.tracking import IoUTracker

logger = logging.getLogger(__name__)


def process_video(
    input_path: str,
    output_path: str,
    bundle_dir: str,
    selected_frame_rate: float | None = None,
) -> None:
    """
    Process a video file, executing inference, tracking, and consensus.
    Writes emitted events to a JSONL output file.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0  # fallback

    frame_interval = 1
    if selected_frame_rate and selected_frame_rate > 0 and selected_frame_rate < fps:
        frame_interval = int(fps / selected_frame_rate)

    pipeline = RuntimePipeline(bundle_dir)
    tracker = IoUTracker(iou_threshold=0.30, max_missed_frames=10)
    consensus = ConsensusEngine(
        minimum_supporting_frames=3,
        window_frames=7,
        minimum_weighted_score=2.2,
        duplicate_suppression_seconds=30.0,
        camera_id=input_file.stem,
    )

    # We will use an arbitrary base UTC time and add seconds to it for mock timestamping
    # based on video position, so logic holds up.
    base_time = datetime.now(UTC)

    events_emitted = 0

    with (
        open(output_file, "w") as out_f,
        tqdm(total=total_frames, desc=f"Processing {input_file.name}") as pbar,
    ):
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                current_time = base_time + timedelta(seconds=(frame_idx / fps))

                # 1. Pipeline (Detector -> Quality -> Orientation -> Transform -> OCR)
                res = pipeline.process_frame(frame)

                # 2. Track
                tracks = tracker.update(res.get("detections", []), current_time)

                # 3. Consensus
                for track in tracks:
                    event = consensus.process_track(track, current_time)
                    if event:
                        out_f.write(event.model_dump_json() + "\n")
                        out_f.flush()
                        events_emitted += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    logger.info(
        f"Video processing complete. Emitted {events_emitted} events to {output_file}"
    )
