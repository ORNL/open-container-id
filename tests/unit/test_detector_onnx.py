import json
import numpy as np
from container_id.runtime.detector_onnx import ONNXDetector


def test_detector_onnx_unscale_boxes(tmp_path):
    # Create mock bundle structure
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text(
        json.dumps({"detector": {"input_shape": [1, 3, 512, 512]}})
    )
    (bundle_dir / "detector_labels.json").write_text(json.dumps(["container_number"]))
    (bundle_dir / "detector.onnx").write_text("mock_onnx_model_data_here")

    detector = ONNXDetector(bundle_dir)
    assert detector._is_mock == True

    # Test unscaling logic
    meta = {"scale": 0.5, "pad_left": 0, "pad_top": 128, "orig_h": 512, "orig_w": 1024}

    # Mock boxes [x1, y1, x2, y2]
    boxes_padded = np.array([[100, 150, 200, 200]], dtype=np.float32)

    unscaled = detector._unscale_boxes(boxes_padded, meta)

    # Expected:
    # x1 = (100 - 0) / 0.5 = 200
    # y1 = (150 - 128) / 0.5 = 44
    # x2 = (200 - 0) / 0.5 = 400
    # y2 = (200 - 128) / 0.5 = 144

    assert np.allclose(unscaled[0], [200, 44, 400, 144])


def test_detector_onnx_mock_inference(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text(
        json.dumps({"detector": {"input_shape": [1, 3, 512, 512]}})
    )
    (bundle_dir / "detector_labels.json").write_text(json.dumps(["container_number"]))
    (bundle_dir / "detector.onnx").write_text("mock_onnx_model_data_here")

    detector = ONNXDetector(bundle_dir)

    # Dummy image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect([img])
    assert len(detections) == 1
    assert len(detections[0]) == 1

    det = detections[0][0]
    assert det.class_name == "container_number"
    assert det.confidence == 0.95
