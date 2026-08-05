from pathlib import Path

import numpy as np
from PIL import Image

from container_id.data.dedupe import (
    deduplicate_dataset,
    find_exact_duplicates,
    find_near_duplicates,
)


def create_image(path: Path, color: str = "white") -> None:
    img = Image.new("RGB", (100, 100), color=color)
    img.save(path)


def create_random_image(path: Path) -> None:
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def test_find_exact_duplicates(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"

    # img1 and img2 are identical, img3 is different
    create_image(img1, "white")
    create_image(img2, "white")
    create_random_image(img3)

    groups = find_exact_duplicates([img1, img2, img3])
    assert len(groups) == 1
    assert len(groups[0].members) == 2
    assert str(img1) in groups[0].members
    assert str(img2) in groups[0].members
    assert groups[0].reason == "exact_hash"


def test_find_near_duplicates(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img4 = tmp_path / "img4.jpg"

    create_image(img1, "white")
    create_image(img2, "white")
    create_image(img3, "white")
    # A completely random image will have a very different phash
    create_random_image(img4)

    # White images will cluster
    groups = find_near_duplicates([img1, img2, img3, img4])
    # Expect 1 group of 3
    assert len(groups) == 1
    assert len(groups[0].members) == 3
    assert str(img1) in groups[0].members
    assert str(img2) in groups[0].members
    assert str(img3) in groups[0].members
    assert str(img4) not in groups[0].members
    assert groups[0].reason == "phash_near_duplicate"


def test_deduplicate_dataset(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"

    create_image(img1, "blue")
    create_image(img2, "blue")
    create_random_image(img3)

    manifest = deduplicate_dataset("test_ds", [img1, img2, img3])

    assert manifest.dataset_id == "test_ds"
    # It will find 1 exact group and 1 phash group containing the identical images
    assert len(manifest.groups) == 2
