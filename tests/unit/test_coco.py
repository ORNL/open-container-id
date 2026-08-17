import json
from pathlib import Path

import cv2
import numpy as np

from container_id.data.coco import discover_coco_splits, validate_coco_split


def create_dummy_image(path: Path, width: int, height: int) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_discover_coco_splits(tmp_path: Path) -> None:
    # Setup standard layout
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "_annotations.coco.json").touch()

    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "_annotations.coco.json").touch()

    # Val alias
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "other.json").touch()

    splits = discover_coco_splits(tmp_path)
    assert "train" in splits
    assert "valid" in splits
    # val maps to valid according to rules
    # It might overwrite if there's both valid and val, but test validates discovery finds them.
    assert len(splits) >= 2


def test_validate_coco_split_valid(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    # Create image
    img_name = "test.jpg"
    img_path = split_dir / img_name
    create_dummy_image(img_path, 100, 100)

    # Create JSON
    json_path = split_dir / "_annotations.coco.json"
    data = {
        "images": [{"id": 1, "file_name": img_name, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
            }
        ],
        "categories": [{"id": 1, "name": "container_number"}],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    report = validate_coco_split(json_path, "train")
    assert report.is_valid()
    assert len(report.errors) == 0


def test_validate_coco_split_missing_image(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    json_path = split_dir / "_annotations.coco.json"
    data = {
        "images": [{"id": 1, "file_name": "missing.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    report = validate_coco_split(json_path, "train")
    assert not report.is_valid()
    assert any("Referenced image does not exist" in e for e in report.errors)


def test_validate_coco_split_invalid_bbox(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    img_name = "test.jpg"
    img_path = split_dir / img_name
    create_dummy_image(img_path, 100, 100)

    json_path = split_dir / "_annotations.coco.json"
    data = {
        "images": [{"id": 1, "file_name": img_name, "width": 100, "height": 100}],
        # bbox out of bounds and negative area
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [150, 150, 50, 50],
                "area": -1,
            }
        ],
        "categories": [{"id": 1, "name": "container_number"}],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    report = validate_coco_split(json_path, "train")
    assert not report.is_valid()
    assert any("entirely outside image bounds" in e for e in report.errors)
    assert any("zero or negative area" in e for e in report.errors)
