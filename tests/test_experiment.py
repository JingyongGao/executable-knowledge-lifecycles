from __future__ import annotations

import pytest
import json
from pathlib import Path

from agent_dynamics.experiment import run_experiment

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def results() -> list[dict[str, object]]:
    return run_experiment(epochs=220, train_size=512, ood_size=512)["results"]


def test_correct_knowledge_improves_ood_and_invariance(results: list[dict[str, object]]) -> None:
    no_k, correct = results[0], results[1]
    assert correct["ood_mse"] < no_k["ood_mse"]
    assert correct["invariance_error"] < no_k["invariance_error"]
    vector = correct["evaluation_vector"]
    assert vector["delta_e_ood_absolute"] > 0
    assert 0 < vector["r_ood_relative_improvement"] <= 1.0
    assert vector["c_infer_ratio"] == 1.0


def test_wrong_knowledge_has_sensitivity_failure(results: list[dict[str, object]]) -> None:
    no_k, _, wrong = results[:3]
    assert wrong["sensitivity_error"] > no_k["sensitivity_error"]
    assert wrong["ood_mse"] > no_k["ood_mse"]


def test_partial_knowledge_leaves_more_shortcut_dependence(results: list[dict[str, object]]) -> None:
    correct, partial = results[1], results[3]
    assert partial["invariance_error"] > correct["invariance_error"]
    assert partial["ood_mse"] > correct["ood_mse"]


def test_stale_knowledge_is_exact_no_k_negative_control(results: list[dict[str, object]]) -> None:
    no_k, stale = results[0], results[4]
    assert stale["ood_mse"] == pytest.approx(no_k["ood_mse"], abs=0.0)
    assert stale["feature_gradients"] == pytest.approx(no_k["feature_gradients"], abs=0.0)


def test_fault_injected_stale_knowledge_causes_material_harm(
    results: list[dict[str, object]],
) -> None:
    no_k, fault = results[0], results[5]
    assert fault["ood_mse"] > no_k["ood_mse"]
    assert fault["evaluation_vector"]["delta_e_ood_absolute"] < 0
    assert fault["evaluation_vector"]["r_ood_relative_improvement"] < 0


def test_frozen_1c_artifact_uses_normalized_metric_dictionary() -> None:
    report = json.loads(
        (ROOT / "artifacts" / "1c_evaluation.json").read_text(encoding="utf-8")
    )
    assert report["spec_version"] == "v0.2.6-exp.1-final"
    for result in report["results"]:
        vector = result["evaluation_vector"]
        expected_delta = vector["e_ood_base_raw"] - vector["e_ood_group_raw"]
        assert vector["delta_e_ood_absolute"] == pytest.approx(expected_delta)
        assert vector["r_ood_relative_improvement"] <= 1.0
        assert vector["c_infer_ratio"] == 1.0
