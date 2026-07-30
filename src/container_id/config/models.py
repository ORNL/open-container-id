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
