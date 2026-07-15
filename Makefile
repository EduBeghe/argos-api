.PHONY: install run test lint format typecheck docker

install:
	uv sync

run:
	uv run uvicorn main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

docker:
	docker compose up --build
