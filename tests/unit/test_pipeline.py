import json

import cv2
import numpy as np

from container_id.runtime.pipeline import RuntimePipeline


def test_pipeline_infer_image_mock(tmp_path):
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

    pipeline = RuntimePipeline(bundle_dir)

    # Create fake image to infer
    img_path = tmp_path / "fake_img.jpg"
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    out_path = tmp_path / "res.json"
    result = pipeline.infer_image(str(img_path), str(out_path))

    assert out_path.exists()
    assert result["model_bundle_version"] == "0.1.0"
    assert len(result["detections"]) == 1

    det = result["detections"][0]
    assert det["detector_confidence"] == 0.95
    assert "best_candidate" in det
    assert det["best_candidate"]["raw_text"] == "BMOU4445146"


def test_pipeline_infer_directory_mock(tmp_path):
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

    pipeline = RuntimePipeline(bundle_dir)

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    cv2.imwrite(str(img_dir / "1.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(img_dir / "2.png"), np.zeros((100, 100, 3), dtype=np.uint8))

    out_jsonl = tmp_path / "results.jsonl"
    pipeline.infer_directory(str(img_dir), str(out_jsonl))

    assert out_jsonl.exists()

    with open(out_jsonl, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    res1 = json.loads(lines[0])
    res2 = json.loads(lines[1])

    assert res1["file"] == "1.jpg"
    assert res2["file"] == "2.png"
