import typer

from container_id.export.detector import export_detector

app = typer.Typer(help="Export commands.")


@app.command("detector")
def cli_export_detector(
    run_dir: str = typer.Option(..., help="Path to the training run directory."),
) -> None:
    """Export the detector model to ONNX."""
    try:
        export_detector(run_dir)
        typer.echo(f"Detector export finished. Results saved in {run_dir}/exports/")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Detector export failed: {e}", err=True)
        raise typer.Exit(1)


from container_id.export.recognizer import export_recognizer


@app.command("ocr")
def cli_export_ocr(
    run_dir: str = typer.Option(..., help="Path to the training run directory."),
) -> None:
    """Export the OCR recognizer model to ONNX."""
    try:
        export_recognizer(run_dir)
        typer.echo(f"OCR export finished. Results saved in {run_dir}/exports/")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"OCR export failed: {e}", err=True)
        raise typer.Exit(1)
