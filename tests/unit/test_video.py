import json
import numpy as np
import cv2
from container_id.streams.video import process_video


def test_process_video_mock(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "detector": {"input_shape": [1, 3, 512, 512]},
                "recognizer": {
                    "input_shape": [1, 3, 32, 128],
                    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                },
            }
        )
    )
    (bundle_dir / "detector_labels.json").write_text(json.dumps(["container_number"]))
    (bundle_dir / "recognizer_charset.txt").write_text(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    (bundle_dir / "detector.onnx").write_text("mock_onnx_model_data_here")
    (bundle_dir / "recognizer.onnx").write_text("mock_onnx_model_data_here")

    # Create fake video
    vid_path = tmp_path / "fake_vid.mp4"
    out = cv2.VideoWriter(
        str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (1920, 1080)
    )
    for _ in range(15):  # 15 frames, at 5fps = 3 seconds
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    out_path = tmp_path / "events.jsonl"

    # Run pipeline
    process_video(
        str(vid_path), str(out_path), str(bundle_dir), selected_frame_rate=2.0
    )

    assert out_path.exists()

    # In the mock, the detector always returns a detection, the recognizer always returns "BMOU4445146"
    # Over 15 frames at 5 FPS: downsampling to 2.0 fps means we process ~6 frames.
    # Consensus engine needs 3 frames. It will definitely emit an event.

    with open(out_path, "r") as f:
        lines = f.readlines()

    assert len(lines) >= 1
    event = json.loads(lines[0])

    assert event["container_number"] == "BMOU4445146"
