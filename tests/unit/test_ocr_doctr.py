from container_id.training.ocr_doctr import train_ocr


def test_ocr_doctr_smoke(tmp_path):
    config = {
        "model": {
            "family": "doctr",
            "architecture": "crnn_mobilenet_v3_small",
            "pretrained": False,
            "vocabulary": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "max_length": 11,
            "input_height": 32,
            "input_width": 128,
        },
        "data": {
            "dataset_dir": str(tmp_path / "mock_data"),
            "include_statuses": ["accepted_check_digit_valid"],
        },
        "training": {
            "device": "cpu",
            "seed": 123,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "early_stopping": False,
            "early_stopping_patience": 5,
            "freeze_backbone_epochs": 0,
            "amp": False,
            "tensorboard": False,
        },
        "augmentation": {
            "grayscale_probability": 0.0,
            "color_jitter_probability": 0.0,
            "shadow_probability": 0.0,
            "gaussian_noise_probability": 0.0,
            "gaussian_blur_probability": 0.0,
            "motion_blur_probability": 0.0,
            "perspective_probability": 0.0,
            "jpeg_compression_probability": 0.0,
            "crop_jitter_probability": 0.0,
        },
        "output": {"root": str(tmp_path / "runs")},
    }

    # We expect it to mock train since the dataset dir doesn't exist
    train_ocr(config)

    run_dir = next(iter((tmp_path / "runs").iterdir()))
    assert run_dir.is_dir()

    manifest_file = run_dir / "run_manifest.json"
    assert manifest_file.exists()

    import json

    with open(manifest_file) as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"
