from dataclasses import dataclass


@dataclass
class ContainerID:
    """Represents a structured container identifier (e.g. ISO 6346)."""

    owner_code: str
    category_id: str
    serial_number: str
    check_digit: int | None = None

    def __str__(self) -> str:
        """Returns the full normalized container ID string."""
        check = str(self.check_digit) if self.check_digit is not None else ""
        return f"{self.owner_code}{self.category_id}{self.serial_number}{check}"

    @property
    def is_valid_length(self) -> bool:
        """Checks if the component parts are the correct length for ISO 6346."""
        return (
            len(self.owner_code) == 3
            and len(self.category_id) == 1
            and len(self.serial_number) == 6
            and (self.check_digit is None or 0 <= self.check_digit <= 9)
        )
