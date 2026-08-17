import json

from container_id.evaluation.detector import evaluate_detector


def test_evaluate_detector_mock(tmp_path):
    run_dir = tmp_path / "runs" / "rfdetr-small-123"
    run_dir.mkdir(parents=True)

    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"run_id": "rfdetr-small-123", "status": "completed"}, f)

    config_path = run_dir / "config.resolved.yaml"
    with open(config_path, "w") as f:
        f.write("data:\n  dataset_dir: fake_dataset_dir\n")

    evaluate_detector(str(run_dir))

    eval_dir = run_dir / "evaluation" / "detector"
    assert eval_dir.exists()
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "threshold_sweep.csv").exists()
    assert (eval_dir / "metrics_by_source.csv").exists()
    assert (eval_dir / "metrics_by_box_size.csv").exists()
    assert (eval_dir / "metrics_by_orientation.csv").exists()
    assert (eval_dir / "report.md").exists()
    assert (eval_dir / "errors_false_negative").is_dir()
    assert (eval_dir / "errors_false_positive").is_dir()
    assert (eval_dir / "contact_sheets").is_dir()

    with open(eval_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
        assert "precision" in metrics
        assert "recall" in metrics
