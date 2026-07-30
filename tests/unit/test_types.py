from container_id.config.models import ContainerIDConfig
from container_id.iso6346.types import ContainerID


def test_container_id_string_representation() -> None:
    cid = ContainerID("CSQ", "U", "305438", 3)
    assert str(cid) == "CSQU3054383"

def test_container_id_no_check_digit() -> None:
    cid = ContainerID("ABC", "U", "123456")
    assert str(cid) == "ABCU123456"

def test_container_id_valid_length() -> None:
    valid_cid = ContainerID("ABC", "U", "123456", 0)
    assert valid_cid.is_valid_length

    invalid_owner = ContainerID("AB", "U", "123456", 0)
    assert not invalid_owner.is_valid_length

def test_config_model() -> None:
    config = ContainerIDConfig(max_correction_edits=1)
    assert config.max_correction_edits == 1
    assert config.validate_check_digit is True
