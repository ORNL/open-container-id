import datetime
from pathlib import Path

from pydantic import BaseModel

from container_id.config.models import DataSourcesConfig
from container_id.util.hashing import calculate_file_sha256


class RegisteredArchive(BaseModel):
    name: str
    absolute_path: str
    filename: str
    byte_size: int
    sha256: str
    registration_utc: str
    url: str | None
    version: str | None
    license: str | None
    attribution: str | None


class SourceRegistry(BaseModel):
    archives: list[RegisteredArchive] = []


def register_sources(config: DataSourcesConfig, registry_path: Path) -> SourceRegistry:
    """Registers dataset archives based on configuration."""
    registry = SourceRegistry()

    for archive_config in config.archives:
        archive_path = Path(archive_config.path).resolve()

        if not archive_path.exists() or not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        byte_size = archive_path.stat().st_size
        sha256 = calculate_file_sha256(archive_path)
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        registered = RegisteredArchive(
            name=archive_config.name,
            absolute_path=str(archive_path),
            filename=archive_path.name,
            byte_size=byte_size,
            sha256=sha256,
            registration_utc=timestamp,
            url=archive_config.url,
            version=archive_config.version,
            license=archive_config.license,
            attribution=archive_config.attribution
        )
        registry.archives.append(registered)

    # Ensure directory exists
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with open(registry_path, "w") as f:
        f.write(registry.model_dump_json(indent=2))

    return registry
