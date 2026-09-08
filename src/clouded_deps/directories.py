"""Project directory paths, resolved relative to the repo root via pathlib."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PIPELINE_DIR = ROOT / "pipeline"
TESTS_DIR = ROOT / "tests"
OUTPUTS_DIR = ROOT / "outputs_results_v2"
