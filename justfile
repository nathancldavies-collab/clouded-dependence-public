install:
    uv sync

lint *files=".":
    uv run ruff check --fix {{files}}
    uv run ruff format {{files}}

pipeline:
    uv run run_pipeline.py