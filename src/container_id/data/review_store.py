import csv
import datetime
import uuid
from pathlib import Path

from container_id.data.schemas import ReviewRecord


class ReviewStore:
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, ReviewRecord] = self._load()

    def _load(self) -> dict[str, ReviewRecord]:
        records = {}
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = ReviewRecord.model_validate_json(line)
                    records[record.review_id] = record
        return records

    def save(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            f.writelines(
                record.model_dump_json() + "\n" for record in self.records.values()
            )

    def add_or_update(self, record: ReviewRecord) -> None:
        if not record.review_id:
            record.review_id = str(uuid.uuid4())
        self.records[record.review_id] = record

    def get_pending(self) -> list[ReviewRecord]:
        return [r for r in self.records.values() if r.decision is None]

    def export_csv(self, export_path: Path) -> None:
        """Exports pending and all records to CSV for manual review."""
        if not self.records:
            return

        fields = list(ReviewRecord.model_fields.keys())
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in self.records.values():
                # dump handles None serialization decently, but let's be explicit
                row = record.model_dump(exclude_none=False)
                writer.writerow(row)

    def import_csv(
        self, import_path: Path, reviewer_name: str = "local_reviewer"
    ) -> int:
        """Imports review decisions from a CSV back into the store."""
        if not import_path.exists():
            return 0

        updated_count = 0
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()

        with open(import_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_id = row.get("review_id")
                if not review_id or review_id not in self.records:
                    continue

                decision = row.get("decision", "").strip()
                reviewed_label = row.get("reviewed_label", "").strip()

                if decision:
                    record = self.records[review_id]
                    if (
                        record.decision != decision
                        or record.reviewed_label != reviewed_label
                    ):
                        record.decision = decision
                        record.reviewed_label = (
                            reviewed_label if reviewed_label else None
                        )
                        record.reviewer = reviewer_name
                        record.reviewed_at_utc = now_utc
                        record.notes = row.get("notes", "").strip() or None
                        updated_count += 1

        if updated_count > 0:
            self.save()

        return updated_count
