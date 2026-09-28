install:
    uv sync

lint *files=".":
    uv run ruff check --fix {{files}}
    uv run ruff format {{files}}

pipeline:
    uv run run_pipeline.py

generate-interrater *args:
    uv run validation/build_validation_sample.py --rater ND JR {{args}}

exposure *args:
    uv run run_exposure_analysis.py {{args}}

plot-exposure *args="--file-type=pdf --scope=entity":
    uv run plot_exposure_barbell.py {{args}}
    uv run plot_ego_networks.py {{args}}
