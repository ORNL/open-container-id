import subprocess
import sys


def test_offline_smoke():
    # Verify the CLI can run without network access
    result = subprocess.run(
        [sys.executable, "-m", "container_id.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={"http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"},
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
