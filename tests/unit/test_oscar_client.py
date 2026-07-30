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
        osh_path_root="/api/sensorhub"
    )

def test_get_alarming_occupancies(oscar_config: OscarConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    client = OscarClient(oscar_config)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "observations": [
            {
                "occupancyObsId": "obs-1",
                "gammaAlarm": True,
                "videoPaths": ["video1.mp4"],
                "controlStreamId": "stream-1"
            },
            {
                "occupancyObsId": "obs-2",
                "gammaAlarm": False,
                "neutronAlarm": False,
                "videoPaths": ["video2.mp4"]
            },
            {
                "occupancyObsId": "obs-3",
                "neutronAlarm": True,
                "videoPaths": ["video3.mp4"]
            }
        ]
    }

    monkeypatch.setattr(client.client, "get", lambda url: mock_response)

    alarming = client.get_alarming_occupancies()

    assert len(alarming) == 2
    assert alarming[0]["occupancyObsId"] == "obs-1"
    assert alarming[1]["occupancyObsId"] == "obs-3"
    assert alarming[0]["controlStreamId"] == "stream-1"
    assert alarming[1]["controlStreamId"] == "test_lane_adjudicationControl"

def test_submit_container_number(oscar_config: OscarConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    client = OscarClient(oscar_config)

    def mock_post(url: str, json: dict) -> MagicMock: # type: ignore[type-arg]
        assert "stream-1" in url
        assert json["parameters"]["vehicleId"] == "MSKU1234567"
        assert json["parameters"]["occupancyObsId"] == "obs-1"
        return MagicMock()

    monkeypatch.setattr(client.client, "post", mock_post)
    client.submit_container_number("stream-1", "obs-1", "MSKU1234567")
