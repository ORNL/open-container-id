from pathlib import Path

from container_id.config.models import CanonicalConfig
from container_id.data.canonical import (
    build_canonical_dataset,
    compute_dataset_fingerprint,
)


def test_build_canonical_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed"
    config = CanonicalConfig(output_dir=str(out_dir))

    summary = build_canonical_dataset(config)
    assert summary["status"] == "success"

    assert (out_dir / "train" / "_annotations.coco.json").exists()
    assert (out_dir / "valid" / "_annotations.coco.json").exists()
    assert (out_dir / "test" / "_annotations.coco.json").exists()

    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "DATASET_NOTICE.md").exists()

    fp = compute_dataset_fingerprint(out_dir)
    assert fp is not None
    assert len(fp) == 64  # sha256 hex length
