import pytest

from container_id.iso6346.check_digit import calculate_check_digit, validate_check_digit


def test_calculate_check_digit_valid() -> None:
    # Example from BIC website: CSQU3054383 -> check digit is 3
    assert calculate_check_digit("CSQU305438") == 3
    # Example from Wikipedia: BICU1234565 -> check digit is 5
    assert calculate_check_digit("BICU123456") == 5


def test_calculate_check_digit_invalid_length() -> None:
    with pytest.raises(ValueError, match="exactly 10 characters"):
        calculate_check_digit("CSQU30543")


def test_calculate_check_digit_invalid_chars() -> None:
    with pytest.raises(ValueError, match="Invalid character"):
        calculate_check_digit("C-QU305438")


def test_validate_check_digit() -> None:
    assert validate_check_digit("CSQU3054383")
    assert not validate_check_digit("CSQU3054384")
    assert validate_check_digit("BICU1234565")


def test_validate_check_digit_invalid_length() -> None:
    assert not validate_check_digit("CSQU305438")


def test_validate_check_digit_invalid_check() -> None:
    assert not validate_check_digit("CSQU305438X")
