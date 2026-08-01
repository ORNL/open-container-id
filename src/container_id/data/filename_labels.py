import csv
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from container_id.data.schemas import CocoImage
from container_id.iso6346.check_digit import calculate_check_digit


class LabelStatus(str, Enum):
    ACCEPTED_CHECK_DIGIT_VALID = "accepted_check_digit_valid"
    CANDIDATE_INVALID_CHECK_DIGIT = "candidate_invalid_check_digit"
    CANDIDATE_NONSTANDARD_EQUIPMENT_CATEGORY = "candidate_nonstandard_equipment_category"
    AMBIGUOUS_MULTIPLE_CANDIDATES = "ambiguous_multiple_candidates"
    NO_CANDIDATE = "no_candidate"
    SOURCE_NAME_MISSING = "source_name_missing"
    MANUAL_OVERRIDE_VALID = "manual_override_valid"
    MANUAL_OVERRIDE_INVALID_BUT_CONFIRMED = "manual_override_invalid_but_confirmed"
    REJECTED = "rejected"

class ParsedLabel(BaseModel):
    normalized_label: str | None
    status: LabelStatus
    raw_source: str

def extract_preferred_filename(image: CocoImage) -> str | None:
    if image.extra:
        extra_name = image.extra.get("name")
        if extra_name and isinstance(extra_name, str) and extra_name.strip():
            return extra_name

        orig_name = image.extra.get("original_name")
        if orig_name and isinstance(orig_name, str) and orig_name.strip():
            return orig_name

    if image.file_name and image.file_name.strip():
        return image.file_name

    return None

def clean_filename(filename: str) -> str:
    base = Path(filename).name
    rf_pattern = re.compile(r"(_[a-zA-Z0-9]+\.rf\.[a-fA-F0-9]+)(\.[a-zA-Z0-9]+)?$")
    base = rf_pattern.sub("", base)

    exts = [".jpg", ".jpeg", ".png", ".webp"]
    for ext in exts:
        if base.lower().endswith(ext):
            base = base[:-len(ext)]

    base = base.upper()
    return base

def parse_filename_label(image: CocoImage) -> ParsedLabel:
    raw_name = extract_preferred_filename(image)
    if not raw_name:
        return ParsedLabel(normalized_label=None, status=LabelStatus.SOURCE_NAME_MISSING, raw_source="")

    cleaned = clean_filename(raw_name)

    # 6. Search for candidate sequences that allow separators between characters.
    # To avoid matching garbage deep in the string, we should look for boundaries.
    # Let's replace separators with empty string.
    # But only inside the sequence?
    # If we just remove all non-alnums:
    normalized = re.sub(r'[^A-Z0-9]', '', cleaned)

    # Require it to be anchored at the START to avoid deep substring matches like OORA2021082
    # e.g. ^[A-Z]{3}[UJZ][0-9]{7}
    # Wait, what if the filename is "1-132746001-OCR-AS-B01-Copy.jpg"?
    # Normalized: "1132746001OCRASB01COPY"
    # What if the filename is "Copy of BMOU4445146.jpg"?
    # Normalized: "COPYOFBMOU4445146"
    # It seems we should allow any position but ensure we don't accidentally match 11 chars out of words.

    # Let's extract words first by splitting on non-alnum.
    # Re-join them and check? No, the user provided `BMOU 444514 6.jpg` -> `BMOU4445146`.
    # If we use `re.sub(r'[^A-Z0-9]', '', cleaned)` then `BMOU4445146` is contiguous.
    # But it also makes `IMGCONCRANEDOORA20210829` contiguous.
    # What if we enforce that the matching 11 characters are either at the start/end or
    # surrounded by what used to be non-alnum?
    # Actually, the simplest fix is to only consider candidates if the *entire* parsed 11 chars
    # are followed by either end-of-string or a digit (like "-1" becoming "1").
    # If it is preceded by a letter, it's probably part of a word.

    pattern = re.compile(r'[A-Z]{3}[UJZ][0-9]{7}')
    candidates = pattern.findall(normalized)

    if len(candidates) == 0:
        nonstd_pattern = re.compile(r'(?<![A-Z])[A-Z]{4}[0-9]{7}')
        nonstd_candidates = nonstd_pattern.findall(normalized)
        if len(nonstd_candidates) == 1:
            return ParsedLabel(
                normalized_label=nonstd_candidates[0],
                status=LabelStatus.CANDIDATE_NONSTANDARD_EQUIPMENT_CATEGORY,
                raw_source=raw_name
            )
        return ParsedLabel(normalized_label=None, status=LabelStatus.NO_CANDIDATE, raw_source=raw_name)

    if len(candidates) > 1:
        if len(set(candidates)) == 1:
            candidates = [candidates[0]]
        else:
            return ParsedLabel(normalized_label=None, status=LabelStatus.AMBIGUOUS_MULTIPLE_CANDIDATES, raw_source=raw_name)

    candidate = candidates[0]

    try:
        body = candidate[:10]
        expected_check = int(candidate[10])
        calculated = calculate_check_digit(body)

        if calculated == expected_check:
            return ParsedLabel(normalized_label=candidate, status=LabelStatus.ACCEPTED_CHECK_DIGIT_VALID, raw_source=raw_name)
        else:
            return ParsedLabel(normalized_label=candidate, status=LabelStatus.CANDIDATE_INVALID_CHECK_DIGIT, raw_source=raw_name)

    except ValueError:
        return ParsedLabel(normalized_label=candidate, status=LabelStatus.NO_CANDIDATE, raw_source=raw_name)

class ManualOverride(BaseModel):
    source_dataset_id: str
    exported_file_name: str
    normalized_label: str
    status: LabelStatus
    reviewer: str
    reviewed_at_utc: str
    notes: str | None = None

def load_manual_overrides(csv_path: Path | str) -> dict[str, ManualOverride]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    overrides = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                status = LabelStatus(row["status"])
            except ValueError:
                status = LabelStatus.REJECTED

            override = ManualOverride(
                source_dataset_id=row["source_dataset_id"],
                exported_file_name=row["exported_file_name"],
                normalized_label=row["normalized_label"],
                status=status,
                reviewer=row["reviewer"],
                reviewed_at_utc=row["reviewed_at_utc"],
                notes=row.get("notes")
            )
            overrides[override.exported_file_name] = override

    return overrides
