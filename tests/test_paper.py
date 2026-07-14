from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_generated_paper_macros_match_frozen_compilation_artifact() -> None:
    report = json.loads(
        (ROOT / "artifacts" / "1a_1b_metrics.json").read_text(encoding="utf-8")
    )
    macros = (ROOT / "paper" / "generated_metrics.tex").read_text(encoding="utf-8")
    assert rf"\newcommand{{\UniqueCases}}{{{report['unique_semantic_cases']}}}" in macros
    assert rf"\newcommand{{\TotalEvents}}{{{report['total_decoding_events']}}}" in macros
    assert r"\newcommand{\SemanticMatchPct}{100.0\%}" in macros
    manifest = json.loads(
        (ROOT / "artifacts" / "release_manifest.json").read_text(encoding="utf-8")
    )
    bundle = ROOT / manifest["bundle"]["path"]
    assert manifest["release_id"] == "v0.2.6-exp.1-final"
    assert manifest["source_control"]["status"] in {
        "local_freeze_only",
        "externally_locatable",
    }
    assert bundle.is_file()
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == manifest["bundle"]["sha256"]


def test_manuscript_states_scope_and_independence_limitations() -> None:
    manuscript = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert "not independent semantic samples" in manuscript
    assert "does not validate the truth" in manuscript
    assert "not a general-purpose measure of distribution shift" in manuscript
    assert "manually authored templates" in manuscript
    assert "severe semantic error" in manuscript
    assert "reverses causal direction" in manuscript
    assert r"\input{generated_release.tex}" in manuscript


def test_generated_execution_table_contains_all_six_conditions() -> None:
    table = (ROOT / "paper" / "generated_results_table.tex").read_text(encoding="utf-8")
    for condition in (
        "No K",
        "Correct K",
        "Wrong K",
        "Partial K",
        "Stale K",
        "Stale K, fault-injected active",
    ):
        assert condition in table
