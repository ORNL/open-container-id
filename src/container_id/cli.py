import typer

app = typer.Typer(help="Open Container ID CLI")

evaluate_app = typer.Typer()
app.add_typer(evaluate_app, name="evaluate", help="Evaluation commands.")

from container_id.export import app as export_app

app.add_typer(export_app, name="export", help="Export commands.")

from container_id.training import app as train_app

app.add_typer(train_app, name="train", help="Training commands.")


@app.callback()
def callback() -> None:
    pass


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo("0.1.0")


if __name__ == "__main__":
    app()


@app.command()
def oscar_poll() -> None:
    """Poll OSCAR for alarming occupancies and submit container numbers."""

    import typer

    from container_id.config.models import OscarConfig
    from container_id.oscar.client import OscarClient

    config = OscarConfig()
    client = OscarClient(config)

    typer.echo("Polling OSCAR for alarming occupancies...")
    try:
        occupancies = client.get_alarming_occupancies()
        for occ in occupancies:
            typer.echo(f"Found alarming occupancy: {occ['occupancyObsId']}")

            # Mock OCR logic here since model pipeline is not yet fully complete
            # We assume it reads MSKU1234567 for testing integration.
            mock_container_number = "MSKU1234567"

            typer.echo(
                f"Submitting container number {mock_container_number} to {occ['controlStreamId']}"
            )
            client.submit_container_number(
                control_stream_id=occ["controlStreamId"],
                occupancy_obs_id=occ["occupancyObsId"],
                container_number=mock_container_number,
            )

    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error polling OSCAR: {e}")
    finally:
        client.close()


from pathlib import Path

import yaml

from container_id.config.models import DataSourcesConfig
from container_id.data.archives import safe_extract_zip
from container_id.data.registry import register_sources

data_app = typer.Typer(help="Data management commands.")
app.add_typer(data_app, name="data")


@data_app.command(name="register")
def register_data(
    config: str = typer.Option(..., help="Path to sources YAML config."),
) -> None:
    """Register dataset archives."""
    config_path = Path(config)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    sources_config = DataSourcesConfig(**yaml_data)

    registry_path = Path("data/manifests/source_registry.local.json")
    try:
        registry = register_sources(sources_config, registry_path)
        typer.echo(
            f"Successfully registered {len(registry.archives)} archives to {registry_path}"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Registration failed: {e}", err=True)


@data_app.command(name="extract")
def extract_data(
    config: str = typer.Option(..., help="Path to sources YAML config."),
    force: bool = typer.Option(
        False, help="Force overwrite of existing extracted directories."
    ),
) -> None:
    """Extract registered dataset archives."""
    config_path = Path(config)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    sources_config = DataSourcesConfig(**yaml_data)

    for archive in sources_config.archives:
        target_dir = Path("data/raw/extracted") / archive.name
        typer.echo(f"Extracting {archive.path} to {target_dir}...")
        try:
            manifest = safe_extract_zip(archive.path, target_dir, force=force)

            manifest_path = (
                Path("data/manifests") / f"{archive.name}_extraction_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as mf:
                mf.write(manifest.model_dump_json(indent=2))

            typer.echo(
                f"Extracted {len(manifest.members)} files. Manifest written to {manifest_path}"
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Extraction failed for {archive.name}: {e}", err=True)


from container_id.data.coco import discover_coco_splits, validate_coco_split


@data_app.command(name="validate-coco")
def validate_coco(
    source: str = typer.Option(..., help="Path to the extracted COCO dataset source."),
) -> None:
    """Validate a COCO dataset structure."""
    source_path = Path(source)
    if not source_path.exists():
        typer.echo(f"Source directory not found: {source_path}", err=True)
        raise typer.Exit(1)

    splits = discover_coco_splits(source_path)
    if not splits:
        typer.echo(f"No COCO JSON files found in {source_path}", err=True)
        raise typer.Exit(1)

    all_valid = True
    for split_name, json_path in splits.items():
        typer.echo(f"\nValidating split: {split_name} ({json_path})")
        report = validate_coco_split(json_path, split_name)

        typer.echo(f"  Images: {report.images_count}")
        typer.echo(f"  Annotations: {report.annotations_count}")
        typer.echo(f"  Categories: {report.categories_count}")

        if report.images_without_annotations > 0:
            typer.echo(
                f"  Note: {report.images_without_annotations} images have no annotations."
            )

        for warning in report.warnings:
            typer.echo(f"  WARNING: {warning}", err=True)

        if report.is_valid():
            typer.echo("  Status: VALID")
        else:
            all_valid = False
            typer.echo("  Status: INVALID")
            for error in report.errors:
                typer.echo(f"  ERROR: {error}", err=True)

    if not all_valid:
        raise typer.Exit(1)


import datetime

from container_id.data.audit import run_dataset_audit
from container_id.data.contact_sheets import generate_contact_sheets


@data_app.command(name="audit")
def audit_data(
    config: str = typer.Option(..., help="Path to sources YAML config."),
) -> None:
    """Audit registered dataset archives and generate contact sheets."""
    config_path = Path(config)
    with open(config_path, "r") as f:
        import yaml

        yaml_data = yaml.safe_load(f)

    sources_config = DataSourcesConfig(**yaml_data)

    utc_run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(f"artifacts/audit/{utc_run_id}")

    typer.echo(f"Running audit. Output will be saved to {output_dir}")
    try:
        run_dataset_audit(sources_config, output_dir)
        generate_contact_sheets(output_dir)
        typer.echo("Audit completed successfully.")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Audit failed: {e}", err=True)
        raise typer.Exit(1)


from container_id.data.audit import acknowledge_audit


@data_app.command(name="acknowledge-audit")
def cli_acknowledge_audit(
    audit_dir: str = typer.Option(..., help="Path to the audit directory."),
    confirm_class: list[str] = typer.Option(
        ...,
        help="Class semantic confirmation in the format dataset_id:source_class=target_class",
    ),
) -> None:
    """Acknowledge an audit by confirming class semantics."""
    audit_path = Path(audit_dir)
    try:
        acknowledge_audit(audit_path, confirm_class)
        typer.echo(
            f"Audit acknowledged successfully. Wrote to {audit_path / 'acknowledgment.json'}"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Failed to acknowledge audit: {e}", err=True)
        raise typer.Exit(1)


from container_id.data.dedupe import deduplicate_dataset


@data_app.command(name="find-duplicates")
def cli_find_duplicates(
    dataset_id: str = typer.Option(..., help="ID of the dataset."),
    extracted_dir: str = typer.Option(
        ..., help="Path to the extracted dataset directory."
    ),
) -> None:
    """Finds exact and near duplicates in an extracted dataset."""
    target_dir = Path(extracted_dir)
    if not target_dir.exists():
        typer.echo(f"Directory not found: {target_dir}", err=True)
        raise typer.Exit(1)

    typer.echo("Scanning for image files...")
    image_paths: list[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(target_dir.rglob(ext))
        image_paths.extend(target_dir.rglob(ext.upper()))

    typer.echo(f"Found {len(image_paths)} images. Running deduplication...")

    try:
        manifest = deduplicate_dataset(dataset_id, image_paths)
        output_path = target_dir / "duplicate_manifest.json"
        with open(output_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        typer.echo(
            f"Found {len(manifest.groups)} duplicate groups. Wrote to {output_path}"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Failed to run deduplication: {e}", err=True)
        raise typer.Exit(1)


from container_id.config.models import CanonicalConfig
from container_id.data.canonical import build_canonical_dataset


@data_app.command(name="build-canonical")
def cli_build_canonical(
    config: str = typer.Option(..., help="Path to canonical YAML config."),
) -> None:
    """Build the canonical one-class COCO detection dataset."""
    config_path = Path(config)

    # Normally we load YAML but we'll mock it if file doesn't exist
    if not config_path.exists():
        typer.echo(
            f"Warning: config not found at {config_path}, using defaults.", err=True
        )
        canonical_config = CanonicalConfig()
    else:
        with open(config_path, "r") as f:
            import yaml

            yaml_data = yaml.safe_load(f)
        canonical_config = CanonicalConfig(**yaml_data)

    try:
        summary = build_canonical_dataset(canonical_config)
        typer.echo(f"Successfully built canonical dataset: {summary['message']}")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Failed to build canonical dataset: {e}", err=True)
        raise typer.Exit(1)


from container_id.config.models import OcrConfig
from container_id.data.ocr_crops import build_ocr_dataset


@data_app.command(name="build-ocr")
def cli_build_ocr(
    config: str = typer.Option(..., help="Path to OCR YAML config."),
) -> None:
    """Build the OCR crop dataset and docTR labels."""
    config_path = Path(config)

    if not config_path.exists():
        typer.echo(
            f"Warning: config not found at {config_path}, using defaults.", err=True
        )
        ocr_config = OcrConfig()
    else:
        with open(config_path, "r") as f:
            import yaml

            yaml_data = yaml.safe_load(f)
        ocr_config = OcrConfig(**yaml_data)

    try:
        summary = build_ocr_dataset(ocr_config)
        typer.echo(f"Successfully built OCR dataset: {summary['message']}")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Failed to build OCR dataset: {e}", err=True)
        raise typer.Exit(1)


import uuid

from container_id.data.review_store import ReviewStore
from container_id.data.schemas import ReviewRecord


@data_app.command(name="review")
def cli_review(
    audit_dir: str = typer.Option(..., help="Path to the audit directory."),
    export_csv: str = typer.Option(None, help="Export current reviews to CSV."),
    import_csv: str = typer.Option(None, help="Import review decisions from CSV."),
    generate_stub: bool = typer.Option(
        False, help="Generate a stub review record for testing."
    ),
) -> None:
    """Manage the manual review store and UI/CSV flow."""
    audit_path = Path(audit_dir)
    store_path = audit_path / "reviews.jsonl"

    store = ReviewStore(store_path)

    if generate_stub:
        stub = ReviewRecord(
            review_id=str(uuid.uuid4()),
            sample_id="stub_sample",
            queue="filename_no_candidate",
            current_status="pending",
            proposed_label=None,
        )
        store.add_or_update(stub)
        store.save()
        typer.echo(f"Added stub review record to {store_path}")

    if export_csv:
        export_path = Path(export_csv)
        store.export_csv(export_path)
        typer.echo(f"Exported {len(store.records)} records to {export_path}")

    if import_csv:
        import_path = Path(import_csv)
        count = store.import_csv(import_path)
        typer.echo(f"Imported and updated {count} review decisions from {import_path}")

    pending = len(store.get_pending())
    typer.echo(
        f"Review store contains {len(store.records)} total records ({pending} pending)."
    )


@app.command(name="doctor")
def cli_doctor() -> None:
    """Run environment health checks, checking MPS availability and path writability."""
    from container_id.training.device import run_doctor

    typer.echo("Running environment doctor...")
    report = run_doctor()

    typer.echo("\n--- Device Info ---")
    for k, v in report["device_info"].items():
        typer.echo(f"{k}: {v}")

    typer.echo("\n--- Checks ---")
    for k, v in report["checks"].items():
        if k == "writable_directories":
            typer.echo("Writable Directories:")
            for p, is_w in v.items():
                typer.echo(f"  {p}: {is_w}")
        else:
            typer.echo(f"{k}: {v}")

    if not report["device_info"].get("mps_available", False) and not report[
        "device_info"
    ].get("cuda_available", False):
        typer.echo(
            "\nWARNING: No hardware acceleration (MPS or CUDA) found. Training will use CPU and be very slow.",
            err=True,
        )







from container_id.evaluation.detector import evaluate_detector


@evaluate_app.command("detector")
def cli_evaluate_detector(
    run_dir: str = typer.Option(..., help="Path to the training run directory."),
):
    """Evaluate the detector model."""
    try:
        evaluate_detector(run_dir)
        typer.echo(
            f"Detector evaluation finished. Results saved in {run_dir}/evaluation/detector/"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Detector evaluation failed: {e}", err=True)
        raise typer.Exit(1)


from container_id.evaluation.ocr import evaluate_ocr


@evaluate_app.command("ocr")
def cli_evaluate_ocr(
    run_dir: str = typer.Option(..., help="Path to the training run directory."),
):
    """Evaluate the OCR model."""
    try:
        evaluate_ocr(run_dir)
        typer.echo(
            f"OCR evaluation finished. Results saved in {run_dir}/evaluation/ocr/"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"OCR evaluation failed: {e}", err=True)
        raise typer.Exit(1)
