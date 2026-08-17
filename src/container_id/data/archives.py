import os
import zipfile
from pathlib import Path

from pydantic import BaseModel

from container_id.util.hashing import calculate_file_sha256


class ExtractedMember(BaseModel):
    path: str
    size: int
    sha256: str


class ExtractionManifest(BaseModel):
    archive_name: str
    members: list[ExtractedMember]


def is_safe_path(target_dir: Path, file_path: str) -> bool:
    """Checks if the file_path is safe to extract into target_dir."""
    # Reject absolute paths
    if os.path.isabs(file_path):
        return False

    resolved_target = target_dir.resolve()
    # Ensure resolving the joined path falls within the target directory
    try:
        resolved_file = (target_dir / file_path).resolve()
        if not str(resolved_file).startswith(str(resolved_target)):
            return False
    except Exception:  # noqa: BLE001
        return False

    return True


def safe_extract_zip(
    archive_path: Path | str,
    target_dir: Path | str,
    max_total_size: int = 10 * 1024 * 1024 * 1024,  # 10 GB default
    max_files: int = 100000,
    force: bool = False,
) -> ExtractionManifest:
    """Safely extracts a ZIP archive."""
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Target directory {target_dir} is not empty. Use force=True to overwrite."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    total_size = 0
    file_count = 0
    extracted_members = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            # Only count regular files for limits
            if not member.is_dir():
                file_count += 1
                total_size += member.file_size

                if file_count > max_files:
                    raise ValueError(
                        f"Archive exceeds maximum file count limit of {max_files}."
                    )
                if total_size > max_total_size:
                    raise ValueError(
                        f"Archive exceeds maximum size limit of {max_total_size} bytes."
                    )

            if not is_safe_path(target_dir, member.filename):
                raise ValueError(f"Unsafe path detected in archive: {member.filename}")

            # Safely extract
            zf.extract(member, target_dir)

            if not member.is_dir():
                extracted_file_path = target_dir / member.filename
                file_sha256 = calculate_file_sha256(extracted_file_path)
                extracted_members.append(
                    ExtractedMember(
                        path=member.filename, size=member.file_size, sha256=file_sha256
                    )
                )

    # Write a DO NOT EDIT marker
    with open(target_dir / "DO_NOT_EDIT_SOURCE_DATA.txt", "w") as f:
        f.write(
            "This directory contains auto-extracted source data. Do not modify these files manually."
        )

    return ExtractionManifest(archive_name=archive_path.name, members=extracted_members)
