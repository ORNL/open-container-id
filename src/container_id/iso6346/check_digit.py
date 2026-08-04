def calculate_check_digit(container_number: str) -> int:
    """
    Calculates the ISO 6346 check digit for a 10-character container number.
    Returns the check digit (0-9).
    """
    if len(container_number) != 10:
        raise ValueError(
            f"Container number must be exactly 10 characters for check digit calculation. Got: '{container_number}'"
        )

    char_values = {
        "A": 10,
        "B": 12,
        "C": 13,
        "D": 14,
        "E": 15,
        "F": 16,
        "G": 17,
        "H": 18,
        "I": 19,
        "J": 20,
        "K": 21,
        "L": 23,
        "M": 24,
        "N": 25,
        "O": 26,
        "P": 27,
        "Q": 28,
        "R": 29,
        "S": 30,
        "T": 31,
        "U": 32,
        "V": 34,
        "W": 35,
        "X": 36,
        "Y": 37,
        "Z": 38,
    }

    sum_val = 0
    for i, char in enumerate(container_number):
        if char.isdigit():
            val = int(char)
        elif char in char_values:
            val = char_values[char]
        else:
            raise ValueError(f"Invalid character '{char}' in container number.")

        weight = 2**i
        sum_val += val * weight

    rem = sum_val % 11
    if rem == 10:
        return 0
    return rem


def validate_check_digit(container_number_with_check: str) -> bool:
    """
    Validates a full 11-character container number against its check digit.
    """
    if len(container_number_with_check) != 11:
        return False

    body = container_number_with_check[:10]
    expected_check_digit = container_number_with_check[10]

    if not expected_check_digit.isdigit():
        return False

    try:
        calculated_check_digit = calculate_check_digit(body)
    except ValueError:
        return False

    return str(calculated_check_digit) == expected_check_digit
