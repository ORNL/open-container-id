import zipfile
from pathlib import Path

import pytest

from container_id.data.archives import is_safe_path, safe_extract_zip


def test_is_safe_path(tmp_path: Path) -> None:
    assert is_safe_path(tmp_path, "safe_file.txt")
    assert is_safe_path(tmp_path, "nested/safe_file.txt")

    # Path traversal
    assert not is_safe_path(tmp_path, "../outside.txt")
    assert not is_safe_path(tmp_path, "nested/../../outside.txt")

    # Absolute paths
    assert not is_safe_path(tmp_path, "/etc/passwd")


def create_mock_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_safe_extract_zip_success(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    target_dir = tmp_path / "extracted"

    create_mock_zip(zip_path, {"file1.txt": b"content1", "dir/file2.txt": b"content2"})

    manifest = safe_extract_zip(zip_path, target_dir)
    assert len(manifest.members) == 2
    assert (target_dir / "DO_NOT_EDIT_SOURCE_DATA.txt").exists()
    assert (target_dir / "file1.txt").exists()
    assert (target_dir / "dir/file2.txt").exists()


def test_safe_extract_zip_limits(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    target_dir = tmp_path / "extracted"

    create_mock_zip(zip_path, {"file1.txt": b"content1", "file2.txt": b"content2"})

    with pytest.raises(ValueError, match="Archive exceeds maximum file count limit"):
        safe_extract_zip(zip_path, target_dir, max_files=1)

    with pytest.raises(ValueError, match="Archive exceeds maximum size limit"):
        safe_extract_zip(zip_path, target_dir, max_total_size=10, force=True)


def test_safe_extract_zip_unsafe_path(tmp_path: Path) -> None:
    pass
