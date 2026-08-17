import json

from container_id.export.detector import export_detector


def test_export_detector_mock(tmp_path):
    run_dir = tmp_path / "runs" / "rfdetr-small-123"
    run_dir.mkdir(parents=True)

    export_detector(str(run_dir))

    export_dir = run_dir / "exports"
    assert export_dir.exists()
    assert (export_dir / "detector.onnx").exists()
    assert (export_dir / "manifest.json").exists()

    with open(export_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
        assert manifest["file"] == "detector.onnx"
        assert manifest["family"] == "rfdetr"
