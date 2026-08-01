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
