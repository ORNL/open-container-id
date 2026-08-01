import csv
from pathlib import Path

from container_id.data.filename_labels import (
    LabelStatus,
    load_manual_overrides,
    parse_filename_label,
)
from container_id.data.schemas import CocoImage


def create_image(file_name: str, extra_name: str | None = None) -> CocoImage:
    extra = {"name": extra_name} if extra_name else None
    return CocoImage(id=1, file_name=file_name, width=100, height=100, extra=extra)

def test_parse_filename_label_examples() -> None:
    # BMOU4445146-1-.jpg                    -> BMOU4445146 (Valid)
    img1 = create_image("BMOU4445146-1-.jpg")
    res1 = parse_filename_label(img1)
    assert res1.normalized_label == "BMOU4445146"

    # BMOU4460053-1-Copy.jpg                -> BMOU4460053
    img2 = create_image("BMOU4460053-1-Copy.jpg")
    res2 = parse_filename_label(img2)
    assert res2.normalized_label == "BMOU4460053"

    # BMOU 444514 6.jpg                     -> BMOU4445146
    img3 = create_image("BMOU 444514 6.jpg")
    res3 = parse_filename_label(img3)
    assert res3.normalized_label == "BMOU4445146"

    # BMOU_444514_6_rotated.png             -> BMOU4445146
    img4 = create_image("BMOU_444514_6_rotated.png")
    res4 = parse_filename_label(img4)
    assert res4.normalized_label == "BMOU4445146"

    # IMG_CON_CRANE_DOOR_A_20210829.jpg     -> no_candidate
    img5 = create_image("IMG_CON_CRANE_DOOR_A_20210829.jpg")
    res5 = parse_filename_label(img5)
    assert res5.status == LabelStatus.NO_CANDIDATE

    # IMG129-Copy.jpg                       -> no_candidate
    img6 = create_image("IMG129-Copy.jpg")
    res6 = parse_filename_label(img6)
    assert res6.status == LabelStatus.NO_CANDIDATE

    # 1-132746001-OCR-AS-B01-Copy.jpg       -> no_candidate
    img7 = create_image("1-132746001-OCR-AS-B01-Copy.jpg")
    res7 = parse_filename_label(img7)
    assert res7.status == LabelStatus.NO_CANDIDATE

def test_parse_filename_label_priority() -> None:
    # Use extra.name if provided
    img = create_image("ugly_name.jpg", extra_name="BMOU4445146-1-.jpg")
    res = parse_filename_label(img)
    assert res.normalized_label == "BMOU4445146"

def test_parse_filename_label_roboflow_suffix() -> None:
    img = create_image("BMOU4445146_jpg.rf.28ea7b9f1f561335e364c5fc5d8a9a37.jpg")
    res = parse_filename_label(img)
    assert res.normalized_label == "BMOU4445146"

def test_parse_filename_label_nonstandard_equipment() -> None:
    # Assuming A is a non-standard equipment category if it's the 4th letter
    # Standard is U, J, Z.
    # CSQA3054383
    img = create_image("CSQA3054383.jpg")
    res = parse_filename_label(img)
    assert res.status == LabelStatus.CANDIDATE_NONSTANDARD_EQUIPMENT_CATEGORY
    assert res.normalized_label == "CSQA3054383"

def test_parse_filename_label_invalid_check() -> None:
    # CSQU3054384 is invalid check (valid is 3)
    img = create_image("CSQU3054384.jpg")
    res = parse_filename_label(img)
    assert res.status == LabelStatus.CANDIDATE_INVALID_CHECK_DIGIT
    assert res.normalized_label == "CSQU3054384"

def test_parse_filename_label_ambiguous() -> None:
    # Two valid-looking strings in the same filename
    img = create_image("CSQU3054383_and_HLXU1234561.jpg")
    res = parse_filename_label(img)
    assert res.status == LabelStatus.AMBIGUOUS_MULTIPLE_CANDIDATES

def test_load_manual_overrides(tmp_path: Path) -> None:
    csv_file = tmp_path / "overrides.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_dataset_id", "exported_file_name", "normalized_label", "status", "reviewer", "reviewed_at_utc", "notes"])
        writer.writerow(["ds1", "file1.jpg", "CSQU3054383", "manual_override_valid", "jules", "2023-10-27T10:00:00Z", "looks fine"])
        writer.writerow(["ds1", "file2.jpg", "CSQU3054384", "manual_override_invalid_but_confirmed", "jules", "2023-10-27T10:00:00Z", "verified invalid check digit"])

    overrides = load_manual_overrides(csv_file)
    assert len(overrides) == 2
    assert "file1.jpg" in overrides
    assert overrides["file1.jpg"].status == LabelStatus.MANUAL_OVERRIDE_VALID
    assert overrides["file2.jpg"].status == LabelStatus.MANUAL_OVERRIDE_INVALID_BUT_CONFIRMED
