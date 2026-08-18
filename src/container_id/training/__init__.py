import typer
import yaml

from container_id.training.detector_rfdetr import train_detector

app = typer.Typer()


@app.command("detector")
def detector(config: str = typer.Option(..., help="Path to training config")) -> None:
    """Train the detector model."""
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    train_detector(config_data)


from container_id.training.ocr_doctr import train_ocr


@app.command("ocr")
def ocr(config: str = typer.Option(..., help="Path to OCR training config")) -> None:
    """Train the docTR OCR model."""
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    train_ocr(config_data)


if __name__ == "__main__":
    app()
