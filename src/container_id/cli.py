import typer

app = typer.Typer(help="Open Container ID CLI")

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

            typer.echo(f"Submitting container number {mock_container_number} to {occ['controlStreamId']}")
            client.submit_container_number(
                control_stream_id=occ["controlStreamId"],
                occupancy_obs_id=occ["occupancyObsId"],
                container_number=mock_container_number
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
def register_data(config: str = typer.Option(..., help="Path to sources YAML config.")) -> None:
    """Register dataset archives."""
    config_path = Path(config)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    sources_config = DataSourcesConfig(**yaml_data)

    registry_path = Path("data/manifests/source_registry.local.json")
    try:
        registry = register_sources(sources_config, registry_path)
        typer.echo(f"Successfully registered {len(registry.archives)} archives to {registry_path}")
    except Exception as e: # noqa: BLE001
        typer.echo(f"Registration failed: {e}", err=True)

@data_app.command(name="extract")
def extract_data(
    config: str = typer.Option(..., help="Path to sources YAML config."),
    force: bool = typer.Option(False, help="Force overwrite of existing extracted directories.")
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

            manifest_path = Path("data/manifests") / f"{archive.name}_extraction_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as mf:
                mf.write(manifest.model_dump_json(indent=2))

            typer.echo(f"Extracted {len(manifest.members)} files. Manifest written to {manifest_path}")
        except Exception as e: # noqa: BLE001
            typer.echo(f"Extraction failed for {archive.name}: {e}", err=True)

from container_id.data.coco import discover_coco_splits, validate_coco_split


@data_app.command(name="validate-coco")
def validate_coco(source: str = typer.Option(..., help="Path to the extracted COCO dataset source.")) -> None:
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
            typer.echo(f"  Note: {report.images_without_annotations} images have no annotations.")

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
def audit_data(config: str = typer.Option(..., help="Path to sources YAML config.")) -> None:
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
    except Exception as e: # noqa: BLE001
        typer.echo(f"Audit failed: {e}", err=True)
        raise typer.Exit(1)
