from unittest.mock import MagicMock

import pytest

from container_id.config.models import OscarConfig
from container_id.oscar.client import OscarClient


@pytest.fixture
def oscar_config() -> OscarConfig:
    return OscarConfig(
        endpoint="http://test",
        client_id="test-client",
        api_key="test-key",
        system_id="test_lane",
        osh_path_root="/api/sensorhub",
    )


def test_get_alarming_occupancies(
    oscar_config: OscarConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = OscarClient(oscar_config)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "observations": [
            {
                "occupancyObsId": "obs-1",
                "gammaAlarm": True,
                "videoPaths": ["video1.mp4"],
                "controlStreamId": "stream-1",
            },
            {
                "occupancyObsId": "obs-2",
                "gammaAlarm": False,
                "neutronAlarm": False,
                "videoPaths": ["video2.mp4"],
            },
            {
                "occupancyObsId": "obs-3",
                "neutronAlarm": True,
                "videoPaths": ["video3.mp4"],
            },
        ]
    }

    monkeypatch.setattr(client.client, "get", lambda url: mock_response)

    alarming = client.get_alarming_occupancies()

    assert len(alarming) == 2
    assert alarming[0]["occupancyObsId"] == "obs-1"
    assert alarming[1]["occupancyObsId"] == "obs-3"
    assert alarming[0]["controlStreamId"] == "stream-1"
    assert alarming[1]["controlStreamId"] == "test_lane_adjudicationControl"


def test_submit_container_number(
    oscar_config: OscarConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = OscarClient(oscar_config)

    def mock_post(url: str, json: dict) -> MagicMock:  # type: ignore[type-arg]
        assert "stream-1" in url
        assert json["parameters"]["vehicleId"] == "MSKU1234567"
        assert json["parameters"]["occupancyObsId"] == "obs-1"
        return MagicMock()

    monkeypatch.setattr(client.client, "post", mock_post)
    client.submit_container_number("stream-1", "obs-1", "MSKU1234567")

import json
import os
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from container_id.cli import app


def test_oscar_poll_cli_success(tmp_path: os.PathLike[str]) -> None:
    runner = CliRunner()

    with patch("container_id.oscar.client.OscarClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock alarming occupancies
        mock_client.get_alarming_occupancies.return_value = [
            {
                "occupancyObsId": "obs-123",
                "controlStreamId": "sys_adjudicationControl",
                "videoPaths": ["/path/to/video1.mp4"]
            }
        ]

        # Mock video download
        mock_client.download_video.return_value = b"fake video content"

        with patch("container_id.streams.video.process_video") as mock_process_video:
            def mock_process(input_path: str, output_path: str, bundle_dir: str, **kwargs: Any) -> None:
                with open(output_path, "w") as f:
                    f.write(json.dumps({"container_number": "ABCD1234567"}) + "\n")

            mock_process_video.side_effect = mock_process

            result = runner.invoke(app, ["oscar-poll", "--models", str(tmp_path)])
            assert result.exit_code == 0

            mock_client.submit_container_number.assert_called_once_with(
                control_stream_id="sys_adjudicationControl",
                occupancy_obs_id="obs-123",
                container_number="ABCD1234567"
            )

def test_oscar_poll_cli_no_video(tmp_path: os.PathLike[str]) -> None:
    runner = CliRunner()
    with patch("container_id.oscar.client.OscarClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_alarming_occupancies.return_value = [
            {
                "occupancyObsId": "obs-123",
                "controlStreamId": "sys_adjudicationControl",
                "videoPaths": []
            }
        ]

        result = runner.invoke(app, ["oscar-poll", "--models", str(tmp_path)])
        assert result.exit_code == 0
        mock_client.submit_container_number.assert_not_called()
