def normalize_container_number(raw_string: str) -> str:
    """
    Normalizes a container number string by stripping whitespace and non-alphanumeric characters.
    Converts to uppercase.
    """
    normalized = ""
    for char in raw_string:
        if char.isalnum():
            normalized += char.upper()
    return normalized
