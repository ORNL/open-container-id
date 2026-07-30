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
