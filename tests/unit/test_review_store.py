import csv
from pathlib import Path

from container_id.data.review_store import ReviewStore
from container_id.data.schemas import ReviewRecord


def test_review_store_lifecycle(tmp_path: Path) -> None:
    store_path = tmp_path / "reviews.jsonl"
    store = ReviewStore(store_path)

    r1 = ReviewRecord(
        review_id="r1", sample_id="s1", queue="test", current_status="pending"
    )
    store.add_or_update(r1)
    store.save()

    # Reload
    store2 = ReviewStore(store_path)
    assert len(store2.records) == 1
    assert "r1" in store2.records
    assert len(store2.get_pending()) == 1


def test_review_store_csv_flow(tmp_path: Path) -> None:
    store_path = tmp_path / "reviews.jsonl"
    csv_path = tmp_path / "export.csv"

    store = ReviewStore(store_path)
    r1 = ReviewRecord(
        review_id="r1", sample_id="s1", queue="test", current_status="pending"
    )
    store.add_or_update(r1)

    # Export
    store.export_csv(csv_path)
    assert csv_path.exists()

    # Simulate a human editing the CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["review_id"] == "r1":
                row["decision"] = "approved"
                row["reviewed_label"] = "MSKU1234567"
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Import
    updated = store.import_csv(csv_path)
    assert updated == 1

    assert store.records["r1"].decision == "approved"
    assert store.records["r1"].reviewed_label == "MSKU1234567"
    assert len(store.get_pending()) == 0
