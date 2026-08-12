import json
from container_id.evaluation.ocr import evaluate_ocr

def test_evaluate_ocr_mock(tmp_path):
    run_dir = tmp_path / "runs" / "doctr-crnn-123"
    run_dir.mkdir(parents=True)

    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"run_id": "doctr-crnn-123", "status": "completed"}, f)

    config_path = run_dir / "config.resolved.yaml"
    with open(config_path, "w") as f:
        f.write("data:\n  dataset_dir: fake_dataset_dir\n")

    evaluate_ocr(str(run_dir))

    eval_dir = run_dir / "evaluation" / "ocr"
    assert eval_dir.exists()
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "predictions.jsonl").exists()
    assert (eval_dir / "confusion_matrix.csv").exists()
    assert (eval_dir / "metrics_by_source.csv").exists()
    assert (eval_dir / "metrics_by_orientation.csv").exists()
    assert (eval_dir / "metrics_by_quality.csv").exists()
    assert (eval_dir / "report.md").exists()
    assert (eval_dir / "errors_exact_match").is_dir()
    assert (eval_dir / "errors_invalid_check_digit").is_dir()
    assert (eval_dir / "contact_sheets").is_dir()

    with open(eval_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
        assert "exact_match" in metrics
        assert "character_error_rate" in metrics
