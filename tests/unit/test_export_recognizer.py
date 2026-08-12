import json

from container_id.export.recognizer import export_recognizer


def test_export_recognizer_mock(tmp_path):
    run_dir = tmp_path / "runs" / "doctr-crnn-123"
    run_dir.mkdir(parents=True)

    export_recognizer(str(run_dir))

    export_dir = run_dir / "exports"
    assert export_dir.exists()
    assert (export_dir / "recognizer.onnx").exists()
    assert (export_dir / "manifest.json").exists()

    with open(export_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["file"] == "recognizer.onnx"
        assert manifest["family"] == "doctr"
        assert "charset" in manifest
        assert manifest["max_length"] == 11
        assert "mean" in manifest.get("normalization", {})
