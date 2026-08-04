import typer
import yaml

from container_id.training.detector_rfdetr import train_detector

app = typer.Typer()


@app.command("detector")
def detector(config: str = typer.Option(..., help="Path to training config")):
    """Train the detector model."""
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    train_detector(config_data)


if __name__ == "__main__":
    app()
