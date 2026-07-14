from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_dynamics.stress_test import (
    ConflictClaim,
    LAMBDA_GRID,
    LOW_CONFIDENCE_GRID,
    resolve_conflicts,
    run_activation_fault_pair,
)


ROOT = Path(__file__).parents[1]
METRICS = json.loads(
    (ROOT / "artifacts" / "parameter_sensitivity.json").read_text(encoding="utf-8")
)


def test_equal_confidence_conflict_resolves_to_abstain() -> None:
    resolution = resolve_conflicts(
        (
            ConflictClaim(1, "NON_CAUSAL", 0.75),
            ConflictClaim(1, "CAUSAL", 0.75),
        )
    )
    assert resolution.abstain is True
    assert resolution.non_causal_indices == ()


def test_unequal_confidence_conflict_uses_stronger_claim() -> None:
    resolution = resolve_conflicts(
        (
            ConflictClaim(1, "NON_CAUSAL", 0.8),
            ConflictClaim(1, "CAUSAL", 0.4),
        )
    )
    assert resolution.abstain is False
    assert resolution.non_causal_indices == (1,)


def test_stale_claim_is_safe_until_activation_monitor_is_bypassed() -> None:
    result = run_activation_fault_pair(epochs=220)
    assert result["safe_stale_ood_mse"] == pytest.approx(
        result["no_k_ood_mse"], abs=0.0
    )
    assert result["fault_injected_ood_mse"] > result["safe_stale_ood_mse"]
    assert result["generalization_mse_drop"] > 1.0


def test_registered_parameter_matrix_has_paired_n30_intervals() -> None:
    assert METRICS["statistical_registration"]["paired_runs"] >= 30
    for knowledge in ("correct_k", "wrong_k"):
        points = METRICS["lambda_sensitivity"][knowledge]
        assert [point["parameter"] for point in points] == list(LAMBDA_GRID)
        assert all(
            point["delta_e_ood_absolute_paired_ci95"]["n"] >= 30
            for point in points
        )


def test_wrong_k_strictly_degrades_and_lambda10_collapses() -> None:
    points = METRICS["lambda_sensitivity"]["wrong_k"]
    errors = [point["ood_mse_mean"] for point in points]
    assert all(right > left for left, right in zip(errors, errors[1:]))
    assert METRICS["lambda10_wrong_k_harm_ci95"]["lower"] > 1.0


def test_correct_k_lambda_is_validation_selected_with_stable_ood_gain() -> None:
    sensitivity = METRICS["lambda_sensitivity"]
    selected = sensitivity["validation_selected_lambda"]
    validation_best = min(
        sensitivity["correct_k"], key=lambda point: point["validation_mse_mean"]
    )["parameter"]
    assert selected == validation_best
    assert len(sensitivity["widest_stable_gain_interval"]) >= 3
    assert METRICS["validations"]["selected_lambda_has_ood_gain_gt_0_1"]


def test_wrong_k_harm_is_monotone_in_confidence() -> None:
    harms = METRICS["confidence_monotonicity"]["wrong_k_harm_means"]
    assert all(right > left for left, right in zip(harms, harms[1:]))
    low = METRICS["confidence_monotonicity"]
    assert low["low_confidence_grid"] == list(LOW_CONFIDENCE_GRID)
    low_harms = low["low_confidence_wrong_k_harm_means"]
    assert all(right > left for left, right in zip(low_harms, low_harms[1:]))
    assert all(
        point["delta_e_ood_absolute_paired_ci95"]["n"] >= 30
        for point in low["low_confidence_wrong_k"]
    )


def test_fault_injection_reports_expected_harm_and_violation_rate() -> None:
    fault = METRICS["fault_injection"]
    assert fault["generalization_mse_drop_ci95"]["lower"] > 1.0
    assert fault["activation_constraint_violation_rate"] == 1.0
    assert fault["harm_incidence_rate"] == 1.0
    assert METRICS["experiment_passed"] is True
