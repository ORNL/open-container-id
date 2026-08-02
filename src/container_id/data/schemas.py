from typing import Any

from pydantic import BaseModel, Field


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str | None = None

class CocoImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    extra: dict[str, Any] | None = None

class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    area: float | None = None
    iscrowd: int | None = 0
    segmentation: list[list[float]] | None = None

class CocoDataset(BaseModel):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]

class ClassConfirmation(BaseModel):
    source_dataset_id: str
    source_class_name: str
    target_class_name: str

class AuditAcknowledgment(BaseModel):
    audit_dir: str
    audit_hash: str
    confirmations: list[ClassConfirmation]
    acknowledged_at_utc: str

class DuplicateGroup(BaseModel):
    group_id: str
    members: list[str] # List of image file paths or image IDs
    reason: str # 'exact_hash', 'phash_near_duplicate', 'source_family'

class DuplicateManifest(BaseModel):
    dataset_id: str
    groups: list[DuplicateGroup]

class CanonicalAnnotation(BaseModel):
    source_annotation_id: int
    source_category: str
    canonical_category: str
    bbox_xywh: list[float]
    bbox_area: float
    orientation_class: str
    review_status: str

class CanonicalSample(BaseModel):
    schema_version: int = 1
    sample_id: str
    source_dataset_id: str
    source_split: str
    source_image_id: int
    exported_file_name: str
    original_file_name: str
    source_image_path: str
    image_sha256: str
    perceptual_hash: str | None = None
    width: int
    height: int
    raw_filename_label: str
    normalized_label: str | None
    label_status: str
    iso_structure_valid: bool
    check_digit_valid: bool
    owner_prefix: str | None
    equipment_category: str | None
    serial_number: str | None
    check_digit: str | None
    annotations: list[CanonicalAnnotation]
    exact_duplicate_group: str | None = None
    near_duplicate_group: str | None = None
    source_family_group: str | None = None
    split_group: str | None = None
    canonical_split: str | None = None
    license: str | None = None

class ReviewRecord(BaseModel):
    review_id: str
    sample_id: str
    annotation_id: str | None = None
    queue: str
    current_status: str
    proposed_label: str | None = None
    reviewed_label: str | None = None
    decision: str | None = None
    reviewer: str | None = None
    reviewed_at_utc: str | None = None
    notes: str | None = None
    audit_run_id: str | None = None
    source_image_sha256: str | None = None

class RunManifest(BaseModel):
    run_id: str
    run_type: str
    started_at_utc: str
    dataset_fingerprint: str | None = None
    config: dict[str, Any] | None = None
    device_info: dict[str, Any] | None = None
    fallback_warnings: list[str] = []
