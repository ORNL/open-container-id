import hashlib
import json
from pathlib import Path

from container_id.config.models import CanonicalConfig
from container_id.data.schemas import (
    CocoCategory,
    CocoDataset,
)


def build_canonical_dataset(config: CanonicalConfig) -> dict[str, str]:
    """Builds the canonical detector dataset from validated sources."""
    # Stub logic for building canonical dataset.
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_dir = out_dir / split
        split_dir.mkdir(exist_ok=True)
        # Write empty COCO stub
        empty_coco = CocoDataset(
            images=[],
            annotations=[],
            categories=[
                CocoCategory(id=0, name="container_number", supercategory="container")
            ],
        )
        with open(split_dir / "_annotations.coco.json", "w") as f:
            f.write(empty_coco.model_dump_json(indent=2))

    # Write empty manifests
    (out_dir / "manifest.jsonl").touch()
    (out_dir / "split_manifest.jsonl").touch()
    (out_dir / "changes.jsonl").touch()

    summary = {"status": "success", "message": "Stub canonical dataset built."}
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "DATASET_NOTICE.md", "w") as f:
        f.write("# Canonical Dataset Notice\n\nAuto-generated.")

    return summary


def compute_dataset_fingerprint(out_dir: Path) -> str:
    """Computes a SHA-256 fingerprint for the dataset based on manifests."""
    sha256 = hashlib.sha256()

    # In full implementation, we'd hash sorted records.
    # For now, hash the summary.
    summary_path = out_dir / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path, "rb") as f:
            sha256.update(f.read())

    return sha256.hexdigest()
