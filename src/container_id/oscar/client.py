from typing import Any

import httpx

from container_id.config.models import OscarConfig


class OscarClient:
    def __init__(self, config: OscarConfig):
        self.config = config
        self.auth = (self.config.client_id, self.config.api_key)
        self.connected_systems_endpoint = f"{self.config.endpoint}{self.config.osh_path_root}/sos/systems/{self.config.system_id}/datastreams"
        self.client = httpx.Client(auth=self.auth, timeout=30.0)

    def get_alarming_occupancies(self) -> list[dict[str, Any]]:
        """Polls for recent alarming occupancies."""
        # Note: the exact query params depend on the OSCAR API spec.
        # We fetch recent observations and filter them.
        url = self.connected_systems_endpoint
        response = self.client.get(url)
        response.raise_for_status()

        data = response.json()
        alarming_occupancies = []

        # Assuming the API returns a list of streams or observations
        # This parsing logic will need to be adapted based on the exact OSCAR JSON schema.
        # We will mock the extraction for now based on the prompt's instructions.
        for obs in data.get("observations", []):
            if obs.get("gammaAlarm") is True or obs.get("neutronAlarm") is True:
                alarming_occupancies.append(
                    {
                        "occupancyObsId": obs.get("occupancyObsId"),
                        "videoPaths": obs.get("videoPaths", []),
                        "controlStreamId": obs.get(
                            "controlStreamId",
                            f"{self.config.system_id}_adjudicationControl",
                        ),
                    }
                )

        return alarming_occupancies

    def download_video(self, video_path: str) -> bytes:
        """Downloads a video file from the given path."""
        url = f"{self.config.endpoint}{self.config.osh_path_root}/buckets/{video_path}"
        response = self.client.get(url)
        response.raise_for_status()
        return response.content

    def submit_container_number(
        self, control_stream_id: str, occupancy_obs_id: str, container_number: str
    ) -> None:
        """Submits the extracted container number back to OSCAR."""
        url = f"{self.config.endpoint}{self.config.osh_path_root}/sos/controlstreams/{control_stream_id}/commands"

        payload = {
            "parameters": {
                "feedback": "OCR automated extraction",
                "adjudicationCode": 0,
                "isotopesCount": 0,
                "isotopes": [],
                "secondaryInspectionStatus": "NONE",
                "filePathCount": 0,
                "filePaths": [],
                "occupancyObsId": occupancy_obs_id,
                "vehicleId": container_number,
            }
        }

        response = self.client.post(url, json=payload)
        response.raise_for_status()

    def close(self) -> None:
        self.client.close()
