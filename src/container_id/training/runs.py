import datetime
from pathlib import Path
from typing import Any

from container_id.data.schemas import RunManifest
from container_id.training.device import get_device_info


class TrainingRun:
    def __init__(
        self,
        run_dir: Path,
        run_type: str,
        config: dict[str, Any],
        dataset_fingerprint: str | None = None,
        allow_mps_cpu_fallback: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        selected_device, device_info = get_device_info(allow_mps_cpu_fallback)
        self.device = selected_device

        self.manifest = RunManifest(
            run_id=self.run_dir.name,
            run_type=run_type,
            started_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            dataset_fingerprint=dataset_fingerprint,
            config=config,
            device_info=device_info,
            fallback_warnings=[],
        )
        self.save_manifest()

    def add_fallback_warning(self, warning: str) -> None:
        self.manifest.fallback_warnings.append(warning)
        self.save_manifest()

    def save_manifest(self) -> None:
        manifest_path = self.run_dir / "run_manifest.json"
        with open(manifest_path, "w") as f:
            f.write(self.manifest.model_dump_json(indent=2))
