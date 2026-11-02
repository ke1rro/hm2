.PHONY: setup notebook check

setup:
	uv python install 3.14
	uv sync --locked
	uv run --locked pre-commit install

notebook:
	uv run --locked jupyter lab notebooks/

check:
	uv run --locked isort --check-only .
	uv run --locked black --check .
	uv run --locked flake8 src/
	uv run --locked ty check
	uv run --locked pre-commit run --all-files
