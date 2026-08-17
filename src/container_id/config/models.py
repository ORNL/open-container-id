from pydantic import BaseModel, ConfigDict


class ContainerIDConfig(BaseModel):
    """Configuration for container ID processing and validation."""

    model_config = ConfigDict(extra="ignore")

    max_correction_edits: int = 2
    validate_check_digit: bool = True


class OscarConfig(BaseModel):
    """Configuration for OSCAR integration."""

    model_config = ConfigDict(extra="ignore")

    endpoint: str = "http://localhost:8080"
    client_id: str = "api-client"
    api_key: str = ""
    poll_interval_seconds: int = 10
    system_id: str = "default_lane"
    osh_path_root: str = "/api/sensorhub"


class SourceArchiveConfig(BaseModel):
    """Configuration for a single dataset archive source."""

    name: str
    path: str
    url: str | None = None
    version: str | None = None
    license: str | None = None
    attribution: str | None = None


class DataSourcesConfig(BaseModel):
    """Configuration containing multiple source archives."""

    model_config = ConfigDict(extra="ignore")
    archives: list[SourceArchiveConfig]


class CanonicalConfig(BaseModel):
    """Configuration for building the canonical dataset."""

    model_config = ConfigDict(extra="ignore")

    source_registry: str = "data/manifests/source_registry.local.json"
    audit_acknowledgment: str = "artifacts/audit/latest/acknowledgment.json"
    output_dir: str = "data/processed/detection-v1"
    link_mode: str = "copy"  # 'hardlink' or 'copy'
    random_seed: int = 6346
    train_pct: float = 0.8
    valid_pct: float = 0.1
    test_pct: float = 0.1


class OcrConfig(BaseModel):
    """Configuration for building OCR datasets."""

    model_config = ConfigDict(extra="ignore")

    canonical_dir: str = "data/processed/detection-v1"
    output_dir: str = "data/processed/ocr-v1"
    padding_fraction: float = 0.08
    include_invalid_confirmed: bool = False


class ReconnectConfig(BaseModel):
    initial_delay_seconds: int = 1
    maximum_delay_seconds: int = 30
    multiplier: int = 2
    jitter_fraction: float = 0.20


class CameraConfig(BaseModel):
    id: str = "gate-1"
    url_env: str = "CONTAINER_ID_RTSP_URL"
    transport: str = "tcp"
    selected_frame_rate: float = 5.0
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 10
    reconnect: ReconnectConfig = ReconnectConfig()


class RTSPConfig(BaseModel):
    schema_version: int = 1
    camera: CameraConfig = CameraConfig()
