import json

from container_id.export.bundle import build_bundle, verify_bundle


def test_bundle_build_and_verify(tmp_path):
    # Setup mock files
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_text("mock_onnx_model_data_detector")

    recognizer_path = tmp_path / "recognizer.onnx"
    recognizer_path.write_text("mock_onnx_model_data_recognizer")

    config_path = tmp_path / "default.yaml"
    config_path.write_text("schema_version: 1\nruntime:\n  detector_threshold: 0.25")

    # Mock manifests
    det_manifest = {"class_names": ["container_number"]}
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(det_manifest, f)

    out_dir = tmp_path / "bundle_out"

    # Build
    build_bundle(
        str(detector_path), str(recognizer_path), str(config_path), str(out_dir)
    )

    # Verify outputs exist
    assert (out_dir / "detector.onnx").exists()
    assert (out_dir / "recognizer.onnx").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "detector_labels.json").exists()
    assert (out_dir / "recognizer_charset.txt").exists()
    assert (out_dir / "runtime_defaults.yaml").exists()
    assert (out_dir / "MODEL_CARD.md").exists()
    assert (out_dir / "MODEL_LICENSE").exists()
    assert (out_dir / "DATASET_ATTRIBUTION.md").exists()
    assert (out_dir / "THIRD_PARTY_NOTICES.md").exists()
    assert (out_dir / "SHA256SUMS").exists()

    # Verify
    verify_bundle(str(out_dir))
