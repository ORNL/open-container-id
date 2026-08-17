from pathlib import Path

from container_id.config.models import OcrConfig
from container_id.data.ocr_crops import build_ocr_dataset


def test_build_ocr_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed_ocr"
    config = OcrConfig(output_dir=str(out_dir))

    summary = build_ocr_dataset(config)
    assert summary["status"] == "success"

    for split in ["train", "valid", "test", "field_test"]:
        assert (out_dir / split / "images").exists()
        assert (out_dir / split / "labels.json").exists()

    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "rejected.jsonl").exists()
    assert (out_dir / "DATASET_NOTICE.md").exists()
