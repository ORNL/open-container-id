import os
import platform
import sys
from typing import Any


def get_device_info(allow_mps_cpu_fallback: bool = False) -> tuple[str, dict[str, Any]]:
    """
    Checks available devices, preferring MPS, then CUDA, then CPU.
    Returns (selected_device_string, device_info_dict)
    """
    if allow_mps_cpu_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    info: dict[str, Any] = {
        "machine": platform.machine(),
        "python_version": sys.version,
        "mac_ver": platform.mac_ver()[0] if platform.system() == "Darwin" else None,
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
        info["cuda_available"] = torch.cuda.is_available()

        if info["mps_available"]:
            selected = "mps"
        elif info["cuda_available"]:
            selected = "cuda"
        else:
            selected = "cpu"

    except ImportError:
        info["pytorch_version"] = None
        info["mps_built"] = False
        info["mps_available"] = False
        info["cuda_available"] = False
        selected = "cpu"

    info["selected_device"] = selected
    info["allow_mps_cpu_fallback"] = allow_mps_cpu_fallback
    return selected, info


def run_doctor() -> dict[str, Any]:
    """Runs a series of health checks for the training environment."""
    _selected, info = get_device_info()

    checks = {
        "python_architecture": info["machine"] == "arm64"
        if platform.system() == "Darwin"
        else True,
        "mps_available": info["mps_available"],
    }

    try:
        import av  # noqa: F401

        checks["pyav_available"] = True
    except ImportError:
        checks["pyav_available"] = False

    try:
        import onnxruntime

        checks["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        checks["onnxruntime_providers"] = []

    # Check writable output directories
    paths = ["artifacts/audit", "data/processed", "runs/detector", "runs/ocr", "models"]
    writable = {}
    for p in paths:
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        writable[p] = os.access(path, os.W_OK)

    checks["writable_directories"] = writable
    return {"device_info": info, "checks": checks}


from pathlib import Path
