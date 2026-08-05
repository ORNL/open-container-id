import typer

from container_id.export.detector import export_detector

app = typer.Typer(help="Export commands.")


@app.command("detector")
def cli_export_detector(
    run_dir: str = typer.Option(..., help="Path to the training run directory."),
):
    """Export the detector model to ONNX."""
    try:
        export_detector(run_dir)
        typer.echo(f"Detector export finished. Results saved in {run_dir}/exports/")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Detector export failed: {e}", err=True)
        raise typer.Exit(1)
