from pathlib import Path

from container_id.config.models import DataSourcesConfig, SourceArchiveConfig
from container_id.data.registry import register_sources


def test_register_sources(tmp_path: Path) -> None:
    archive_file = tmp_path / "dataset.zip"
    archive_file.write_bytes(b"dummy zip content")

    registry_file = tmp_path / "registry.json"

    config = DataSourcesConfig(
        archives=[
            SourceArchiveConfig(
                name="test_dataset",
                path=str(archive_file)
            )
        ]
    )

    registry = register_sources(config, registry_file)

    assert registry_file.exists()
    assert len(registry.archives) == 1

    registered = registry.archives[0]
    assert registered.name == "test_dataset"
    assert registered.filename == "dataset.zip"
    assert registered.byte_size == len(b"dummy zip content")
    assert registered.sha256 is not None
