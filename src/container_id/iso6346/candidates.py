from typing import Optional

from container_id.iso6346.types import ContainerID
from container_id.iso6346.normalize import normalize_container_number
from container_id.iso6346.check_digit import calculate_check_digit


def parse_candidate(raw_candidate: str) -> Optional[ContainerID]:
    """
    Parses a normalized or slightly imperfect string into a ContainerID object.
    Returns None if the string cannot be reasonably parsed as a container ID structure.
    """
    normalized = normalize_container_number(raw_candidate)

    if len(normalized) not in (10, 11):
        return None

    owner_code = normalized[:3]
    category_id = normalized[3:4]
    serial_number = normalized[4:10]

    check_digit = None
    if len(normalized) == 11:
        check_char = normalized[10]
        if check_char.isdigit():
            check_digit = int(check_char)
        else:
            return None # invalid check digit character

    # Structural check - owner must be alphabetic
    if not owner_code.isalpha():
        return None

    # Category ID is typically U, J, Z, but we require it to be alphabetic
    if not category_id.isalpha():
        return None

    # Serial number must be numeric
    if not serial_number.isdigit():
        return None

    return ContainerID(
        owner_code=owner_code,
        category_id=category_id,
        serial_number=serial_number,
        check_digit=check_digit
    )


def score_candidate(candidate: ContainerID, require_valid_check_digit: bool = True) -> int:
    """
    Assigns a simple confidence/validity score to a parsed candidate.
    Higher is better. Returns -1 if it's completely invalid.
    """
    if not candidate.is_valid_length:
        return -1

    score = 0

    # 1. Structural matching
    if candidate.category_id in ("U", "J", "Z"):
        score += 10

    # 2. Check digit validation
    if candidate.check_digit is not None:
        try:
            expected = calculate_check_digit(f"{candidate.owner_code}{candidate.category_id}{candidate.serial_number}")
            if expected == candidate.check_digit:
                score += 50
            elif require_valid_check_digit:
                return -1 # Fails hard validation
            else:
                score -= 10
        except ValueError:
            if require_valid_check_digit:
                return -1
    elif require_valid_check_digit:
        return -1

    return score
