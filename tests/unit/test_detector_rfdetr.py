import pytest
from pathlib import Path
from container_id.training.detector_rfdetr import DetectorTrainConfig, train_detector
import yaml

def test_detector_rfdetr_smoke(tmp_path):
    config = {
        "model": {
            "family": "rfdetr",
            "variant": "small",
            "pretrained": False,
            "class_names": ["container_number"]
        },
        "data": {
            "dataset_dir": str(tmp_path / "mock_data"),
            "dataset_manifest": str(tmp_path / "mock_data" / "manifest.jsonl"),
            "split_manifest": str(tmp_path / "mock_data" / "split_manifest.jsonl")
        },
        "training": {
            "device": "cpu",
            "seed": 123,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "learning_rate": 0.001,
            "use_ema": False,
            "early_stopping": False,
            "early_stopping_patience": 5,
            "checkpoint_interval": 1,
            "tensorboard": False,
            "wandb": False,
            "gradient_checkpointing": False,
            "resolution": "default"
        },
        "output": {
            "root": str(tmp_path / "runs")
        }
    }

    # We expect it to mock train since the dataset dir doesn't exist
    train_detector(config)

    run_dir = list((tmp_path / "runs").iterdir())[0]
    assert run_dir.is_dir()

    manifest_file = run_dir / "run_manifest.json"
    assert manifest_file.exists()

    import json
    with open(manifest_file) as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"
