import json
from pathlib import Path
from typing import Any

from container_id.config.models import OcrConfig


def build_ocr_dataset(config: OcrConfig) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test", "field_test"]:
        split_dir = out_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        # Write empty labels.json stub
        with open(split_dir / "labels.json", "w") as f:
            f.write("{}")

    # Write empty manifests
    (out_dir / "manifest.jsonl").touch()
    (out_dir / "rejected.jsonl").touch()

    summary = {"status": "success", "message": "Stub OCR dataset built."}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "DATASET_NOTICE.md", "w") as f:
        f.write("# OCR Dataset Notice\n\nAuto-generated.")

    return summary
