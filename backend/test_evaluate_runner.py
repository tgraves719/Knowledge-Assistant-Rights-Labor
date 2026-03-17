"""Deterministic checks for canonical-vs-legacy evaluation runner tracks."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.evaluate_runner as evaluate_runner


def _has_module(commands: list[list[str]], module_name: str) -> bool:
    for command in commands:
        try:
            module_index = command.index("-m") + 1
        except ValueError:
            continue
        if module_index < len(command) and command[module_index] == module_name:
            return True
    return False


def _test_all_track_excludes_legacy_baselines() -> None:
    commands = evaluate_runner._build_commands("all", "normal", None, 42)
    assert not _has_module(commands, "backend.evaluate"), (
        "Canonical all-track should not include the legacy v1 evaluator."
    )
    assert not _has_module(commands, "backend.evaluate_comprehensive"), (
        "Canonical all-track should not include the legacy v2 evaluator."
    )
    assert not _has_module(commands, "backend.evaluate_multi_contract"), (
        "Canonical all-track should not include the exploratory v2 multi-contract evaluator."
    )
    required = {
        "backend.evaluate_v3",
        "backend.evaluate_role_catalog_integrity",
        "backend.evaluate_retrieval_stage_consistency",
        "backend.evaluate_real_user_regressions",
        "backend.evaluate_miss_record_integrity",
        "backend.evaluate_moa_readiness",
    }
    for token in required:
        assert any(token in command for command in commands), (
            f"Canonical all-track is missing required command token: {token}"
        )


def _test_legacy_baselines_track_preserves_old_suites() -> None:
    commands = evaluate_runner._build_commands("legacy_baselines", "normal", None, 42)
    assert _has_module(commands, "backend.evaluate")
    assert _has_module(commands, "backend.evaluate_comprehensive")
    assert _has_module(commands, "backend.evaluate_multi_contract")


def main() -> None:
    _test_all_track_excludes_legacy_baselines()
    _test_legacy_baselines_track_preserves_old_suites()
    print("[OK] evaluation runner track checks passed")


if __name__ == "__main__":
    main()
