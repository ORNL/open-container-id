from pathlib import Path

from container_id.training.device import get_device_info, run_doctor
from container_id.training.runs import TrainingRun


def test_get_device_info() -> None:
    selected, info = get_device_info(allow_mps_cpu_fallback=True)
    assert selected in ["mps", "cuda", "cpu"]
    assert "machine" in info
    assert info["allow_mps_cpu_fallback"] is True

def test_run_doctor() -> None:
    report = run_doctor()
    assert "device_info" in report
    assert "checks" in report
    assert "writable_directories" in report["checks"]
    assert "python_architecture" in report["checks"]

def test_training_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_abc"
    config = {"epochs": 100, "batch_size": 16}

    run = TrainingRun(
        run_dir=run_dir,
        run_type="detector",
        config=config,
        dataset_fingerprint="abc123hash",
        allow_mps_cpu_fallback=False
    )

    assert run.manifest.run_id == "run_abc"
    assert run.manifest.run_type == "detector"
    assert run.manifest.config["epochs"] == 100
    assert run.manifest.dataset_fingerprint == "abc123hash"
    assert len(run.manifest.fallback_warnings) == 0

    run.add_fallback_warning("MPS out of memory fallback")
    assert len(run.manifest.fallback_warnings) == 1

    assert (run_dir / "run_manifest.json").exists()
