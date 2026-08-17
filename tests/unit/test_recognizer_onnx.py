import json

import numpy as np

from container_id.runtime.recognizer_onnx import ONNXRecognizer


def test_recognizer_onnx_mock_inference(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "recognizer": {
                    "input_shape": [1, 3, 32, 128],
                    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                }
            }
        )
    )
    (bundle_dir / "recognizer_charset.txt").write_text(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    (bundle_dir / "recognizer.onnx").write_text("mock_onnx_model_data_here")

    recognizer = ONNXRecognizer(bundle_dir)
    assert recognizer._is_mock == True

    # Dummy image
    img = np.zeros((100, 300, 3), dtype=np.uint8)

    results = recognizer.recognize([img])
    assert len(results) == 1
    assert len(results[0]) == 1

    cand = results[0][0]
    assert cand.raw_text == "BMOU4445146"
    assert cand.confidence == 0.98


def test_recognizer_ctc_decode(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text(json.dumps({}))
    (bundle_dir / "recognizer_charset.txt").write_text("ABC")
    (bundle_dir / "recognizer.onnx").write_text("mock_onnx_model_data_here")

    recognizer = ONNXRecognizer(bundle_dir)

    # seq_len = 5, num_classes = 4 (blank + ABC)
    logits = np.array(
        [
            [0.1, 0.9, 0.0, 0.0],  # A
            [0.8, 0.1, 0.0, 0.1],  # blank
            [0.0, 0.0, 0.9, 0.1],  # B
            [0.0, 0.0, 0.9, 0.1],  # B (merged by CTC)
            [0.1, 0.0, 0.0, 0.9],  # C
        ],
        dtype=np.float32,
    )

    text, conf = recognizer._ctc_decode(logits)
    assert text == "ABC"
    assert np.isclose(conf, 0.9)
