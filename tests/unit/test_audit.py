from pathlib import Path

from container_id.config.models import DataSourcesConfig
from container_id.data.audit import run_dataset_audit
from container_id.data.contact_sheets import generate_contact_sheets


def test_run_dataset_audit(tmp_path: Path) -> None:
    config = DataSourcesConfig(archives=[])
    report = run_dataset_audit(config, tmp_path)
    assert report["status"] == "success"
    assert (tmp_path / "audit_report.json").exists()
    assert (tmp_path / "audit_report.md").exists()


def test_generate_contact_sheets(tmp_path: Path) -> None:
    generate_contact_sheets(tmp_path)
    assert tmp_path.exists()


def test_acknowledge_audit(tmp_path: Path) -> None:
    import json

    from container_id.data.audit import acknowledge_audit

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    report_file = audit_dir / "audit_report.json"
    report_file.write_text("{}")

    confirmations = ["pranw_container_number_v7:objects=container_number"]
    ack = acknowledge_audit(audit_dir, confirmations)

    assert ack.audit_dir == str(audit_dir)
    assert len(ack.confirmations) == 1
    assert ack.confirmations[0].source_dataset_id == "pranw_container_number_v7"
    assert ack.confirmations[0].source_class_name == "objects"
    assert ack.confirmations[0].target_class_name == "container_number"

    ack_file = audit_dir / "acknowledgment.json"
    assert ack_file.exists()

    with open(ack_file, "r") as f:
        data = json.load(f)
        assert data["audit_dir"] == str(audit_dir)
