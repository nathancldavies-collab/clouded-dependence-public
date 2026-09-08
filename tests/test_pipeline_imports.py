"""Verify the pipeline modules are importable as part of the clouded_deps package.

These tests exercise wiring only - no dataset is read and no stage is executed.
"""

import importlib

import pandas as pd
import pytest

MODULE_EXPORTS = {
    "clouded_deps.pipeline.create_merged_dataset": [
        "load_primes",
        "load_subs",
        "create_merged_dataset",
        "verify_merged_dataset",
        "print_merge_summary",
        "dedupe_primes_by_award",
    ],
    "clouded_deps.pipeline.filter_primes": ["filter_primes"],
    "clouded_deps.pipeline.baseline_merged_hhi": [
        "calculate_contractor_hhi",
        "compare_baseline_approaches",
        "calculate_hhi",
        "classify_hhi",
    ],
    "clouded_deps.pipeline.cloud_classification": [
        "run_stage1",
        "safe_read_csv",
        "synthesize_classification",
    ],
    "clouded_deps.pipeline.platform_attribution": [
        "run_stage2",
        "calculate_platform_hhi",
        "expand_platform_mentions",
    ],
}


@pytest.mark.parametrize(("module_name", "exports"), list(MODULE_EXPORTS.items()))
def test_module_exports(module_name: str, exports: list[str]) -> None:
    module = importlib.import_module(module_name)
    for export in exports:
        assert callable(getattr(module, export)), f"{module_name}.{export}"


def test_run_pipeline_reexports_package_functions() -> None:
    """run_pipeline binds the same objects the package modules define."""
    run_pipeline = importlib.import_module("run_pipeline")
    attribution = importlib.import_module("clouded_deps.pipeline.platform_attribution")
    classification = importlib.import_module(
        "clouded_deps.pipeline.cloud_classification"
    )

    assert run_pipeline.run_stage2 is attribution.run_stage2
    assert run_pipeline.run_stage1 is classification.run_stage1
    assert run_pipeline.cloud_classification is classification
    assert callable(run_pipeline.run_full_pipeline)


def test_no_dynamic_file_imports_left() -> None:
    """The importlib file-path loading hack is gone from the entry points."""
    from clouded_deps.directories import ROOT

    for path in [ROOT / "run_pipeline.py", ROOT / "batch_manager.py"]:
        assert "spec_from_file_location" not in path.read_text(), path


def test_hhi_helpers_work_on_a_tiny_frame() -> None:
    """Smoke test that an imported function actually runs (no I/O)."""
    from clouded_deps.pipeline.baseline_merged_hhi import calculate_hhi, classify_hhi

    hhi = calculate_hhi(pd.Series([50.0, 50.0]))
    assert hhi == pytest.approx(5000.0)
    assert isinstance(classify_hhi(hhi), str)
