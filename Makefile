.PHONY: install lint format typecheck test check

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy

test:
	pytest -q

check: lint typecheck test
