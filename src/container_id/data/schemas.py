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
