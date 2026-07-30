from pydantic import BaseModel, ConfigDict


class ContainerIDConfig(BaseModel):
    """Configuration for container ID processing and validation."""

    model_config = ConfigDict(extra="ignore")

    max_correction_edits: int = 2
    validate_check_digit: bool = True
