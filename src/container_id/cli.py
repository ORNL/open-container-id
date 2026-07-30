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
