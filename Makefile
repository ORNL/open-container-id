setup:
	uv sync --extra dev

setup-train:
	uv sync --extra dev --extra train --extra export

setup-runtime:
	uv sync --extra runtime --extra rtsp --extra api

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

typecheck:
	uv run mypy src tests

offline-test:
	bash scripts/verify_offline.sh
