import json
from pathlib import Path

from container_id.data.schemas import CocoDataset


def discover_coco_splits(source_dir: Path | str) -> dict[str, Path]:
    """
    Discovers COCO annotation files in a source directory.
    Returns a dictionary mapping split names (e.g., 'train', 'valid', 'test') to their JSON file paths.
    """
    source_dir = Path(source_dir)
    splits = {}

    # Common split directory names
    split_aliases = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

    # Find all json files that look like coco annotations
    # typically _annotations.coco.json or similar.
    # We will search recursively.
    json_files = list(source_dir.rglob("*.json"))

    for jf in json_files:
        # Check parent directory name to infer split
        parent_name = jf.parent.name.lower()
        if parent_name in split_aliases:
            split_name = split_aliases[parent_name]
            splits[split_name] = jf

    return splits


import cv2


class CocoValidationReport:
    def __init__(self, split_name: str, json_path: Path):
        self.split_name = split_name
        self.json_path = json_path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.images_count = 0
        self.annotations_count = 0
        self.categories_count = 0
        self.images_without_annotations = 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_coco_split(json_path: Path, split_name: str) -> CocoValidationReport:
    """Validates a single COCO JSON file and its associated images."""
    report = CocoValidationReport(split_name, json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        report.add_error(f"Failed to parse JSON: {e}")
        return report

    try:
        dataset = CocoDataset(**data)
    except Exception as e:  # noqa: BLE001
        report.add_error(f"Schema validation failed: {e}")
        return report

    report.images_count = len(dataset.images)
    report.annotations_count = len(dataset.annotations)
    report.categories_count = len(dataset.categories)

    # Validate categories
    category_ids = set()
    for cat in dataset.categories:
        if cat.id in category_ids:
            report.add_error(f"Duplicate category ID found: {cat.id}")
        category_ids.add(cat.id)

    # Validate images
    image_ids = set()
    image_filenames = set()
    base_dir = json_path.parent

    for img in dataset.images:
        if img.id in image_ids:
            report.add_error(f"Duplicate image ID found: {img.id}")
        image_ids.add(img.id)

        if img.file_name in image_filenames:
            report.add_warning(f"Duplicate image filename found: {img.file_name}")
        image_filenames.add(img.file_name)

        img_path = base_dir / img.file_name
        if not img_path.exists():
            report.add_error(f"Referenced image does not exist: {img.file_name}")
        else:
            # Check image dimensions
            try:
                # Read using cv2 to verify dimensions
                cv_img = cv2.imread(str(img_path))
                if cv_img is None:
                    report.add_error(f"Failed to decode image: {img.file_name}")
                else:
                    h, w = cv_img.shape[:2]
                    if w != img.width or h != img.height:
                        report.add_error(
                            f"Image dimension mismatch for {img.file_name}. Expected {img.width}x{img.height}, got {w}x{h}."
                        )
            except Exception as e:  # noqa: BLE001
                report.add_error(f"Error reading image {img.file_name}: {e}")

    # Validate annotations
    annotation_ids = set()
    annotated_image_ids = set()

    for ann in dataset.annotations:
        if ann.id in annotation_ids:
            report.add_error(f"Duplicate annotation ID found: {ann.id}")
        annotation_ids.add(ann.id)

        if ann.image_id not in image_ids:
            report.add_error(
                f"Annotation {ann.id} references missing image ID: {ann.image_id}"
            )
        else:
            annotated_image_ids.add(ann.image_id)

        if ann.category_id not in category_ids:
            report.add_error(
                f"Annotation {ann.id} references missing category ID: {ann.category_id}"
            )

        # Box validation
        x, y, w, h = ann.bbox
        if w <= 0 or h <= 0:
            report.add_error(
                f"Annotation {ann.id} has invalid dimensions (w={w}, h={h})"
            )

        # Check bounds (assuming image_id is valid, we'd need a lookup table to check bounds properly)
        img_match = next((i for i in dataset.images if i.id == ann.image_id), None)
        if img_match:
            # A box entirely outside the image
            if (
                x >= img_match.width
                or y >= img_match.height
                or (x + w) <= 0
                or (y + h) <= 0
            ):
                report.add_error(
                    f"Annotation {ann.id} box is entirely outside image bounds."
                )
            # A box partially outside
            elif (
                x < 0
                or y < 0
                or (x + w) > img_match.width
                or (y + h) > img_match.height
            ):
                report.add_error(f"Annotation {ann.id} box is partially out of bounds.")

        if ann.area is not None and ann.area <= 0:
            report.add_error(f"Annotation {ann.id} has zero or negative area.")

    report.images_without_annotations = len(image_ids - annotated_image_ids)

    return report
