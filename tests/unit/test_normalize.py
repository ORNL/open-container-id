from container_id.iso6346.normalize import normalize_container_number


def test_normalize_container_number() -> None:
    assert normalize_container_number("CSQU 305438-3") == "CSQU3054383"
    assert normalize_container_number(" csq u 305438 3  ") == "CSQU3054383"
    assert normalize_container_number("CSQU-305438 [3]") == "CSQU3054383"
    assert normalize_container_number("HLXU_123456") == "HLXU123456"
