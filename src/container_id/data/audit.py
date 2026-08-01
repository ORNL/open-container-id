import json
from pathlib import Path
from typing import Any

from container_id.config.models import DataSourcesConfig


def run_dataset_audit(config: DataSourcesConfig, output_dir: Path) -> dict[str, Any]:
    """Runs a dataset audit and generates a report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for actual audit logic
    report = {
        "status": "success",
        "archives_audited": len(config.archives),
        "message": "Audit completed. (Not fully implemented)"
    }

    with open(output_dir / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "audit_report.md", "w") as f:
        f.write("# Dataset Audit Report\n\n")
        f.write(f"Audited {len(config.archives)} archives.\n")

    return report

import datetime

from container_id.data.schemas import AuditAcknowledgment, ClassConfirmation
from container_id.util.hashing import calculate_file_sha256


def acknowledge_audit(audit_dir: Path, confirmations: list[str]) -> AuditAcknowledgment:
    """
    Creates an acknowledgment file for a given audit directory.
    confirmations is a list of strings formatted as 'dataset_id:source_class=target_class'.
    """
    if not audit_dir.exists():
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    audit_report_path = audit_dir / "audit_report.json"
    if not audit_report_path.exists():
        raise FileNotFoundError(f"Audit report not found in directory: {audit_dir}")

    audit_hash = calculate_file_sha256(audit_report_path)

    parsed_confirmations = []
    for conf in confirmations:
        # Expected format: dataset_id:source_class=target_class
        try:
            dataset_id, mapping = conf.split(":")
            source_class, target_class = mapping.split("=")
            parsed_confirmations.append(
                ClassConfirmation(
                    source_dataset_id=dataset_id,
                    source_class_name=source_class,
                    target_class_name=target_class
                )
            )
        except ValueError:
            raise ValueError(f"Invalid confirmation format '{conf}'. Expected format: 'dataset_id:source_class=target_class'")

    ack = AuditAcknowledgment(
        audit_dir=str(audit_dir),
        audit_hash=audit_hash,
        confirmations=parsed_confirmations,
        acknowledged_at_utc=datetime.datetime.now(datetime.UTC).isoformat()
    )

    ack_path = audit_dir / "acknowledgment.json"
    with open(ack_path, "w") as f:
        f.write(ack.model_dump_json(indent=2))

    return ack
