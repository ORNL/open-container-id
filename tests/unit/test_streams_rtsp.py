import pytest
from unittest.mock import MagicMock, patch
from container_id.streams.rtsp import RTSPRunner

@patch("container_id.streams.rtsp.yaml")
def test_rtsp_runner_init(mock_yaml, tmp_path):
    mock_yaml.safe_load.return_value = {
        "camera": {"url": "rtsp://test", "fps": 15},
        "reconnect": {"max_retries": 3, "backoff_factor": 1.0},
    }

    config_file = tmp_path / "config.yaml"
    config_file.touch()
    models_dir = tmp_path / "models"

    with patch("container_id.streams.rtsp.RuntimePipeline") as mock_pipeline:
        runner = RTSPRunner(str(config_file), str(models_dir))
        assert runner.config.camera.url == "rtsp://test"
        assert runner.config.camera.fps == 15
        assert runner.config.reconnect.max_retries == 3
        mock_pipeline.assert_called_once()
