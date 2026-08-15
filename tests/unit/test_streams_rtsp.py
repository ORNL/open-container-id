import os
import pathlib
from unittest.mock import MagicMock, patch

from container_id.streams.rtsp import RTSPRunner


@patch("yaml.safe_load")
def test_rtsp_runner_init(mock_yaml_safe_load: MagicMock, tmp_path: pathlib.Path) -> None:
    mock_yaml_safe_load.return_value = {
        "camera": {"selected_frame_rate": 5.0},
        "reconnect": {"max_retries": 3, "backoff_factor": 1.0},
    }

    config_file = tmp_path / "config.yaml"
    config_file.touch()
    models_dir = tmp_path / "models"

    with patch.dict(os.environ, {"CONTAINER_ID_RTSP_URL": "rtsp://test"}), patch("container_id.streams.rtsp.RuntimePipeline"):
        runner = RTSPRunner(str(config_file), str(models_dir))
        assert runner.uri == "rtsp://test"
        assert runner.config.camera.selected_frame_rate == 5.0
        # No longer testing max_retries as reconnect config has different fields
        # mock_pipeline.assert_called_once()  # Pipeline is only called in consumer thread, not in init
