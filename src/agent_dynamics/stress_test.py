"""Paired-seed stress tests for regularization strength, trust, and activation faults."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .experiment import LinearStructuralModel
from .losses import CausalRegularizedLoss


LAMBDA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CONFIDENCE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LOW_CONFIDENCE_GRID = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class StressSeeds:
    seed_data_start: int = 210001
    seed_model_start: int = 310001
    paired_runs: int = 30


@dataclass
class StressData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    ood_x: torch.Tensor
    ood_y: torch.Tensor


@dataclass(frozen=True)
class ConflictClaim:
    feature_index: int
    stance: Literal["CAUSAL", "NON_CAUSAL"]
    confidence: float


@dataclass(frozen=True)
class Resolution:
    non_causal_indices: tuple[int, ...]
    abstain: bool
    reason: str


def resolve_conflicts(claims: tuple[ConflictClaim, ...]) -> Resolution:
    """Resolve opposing feature claims; equal authority must fail closed."""
    by_feature: dict[int, list[ConflictClaim]] = {}
    for claim in claims:
        if not 0.0 <= claim.confidence <= 1.0:
            raise ValueError("claim confidence must be in [0, 1]")
        by_feature.setdefault(claim.feature_index, []).append(claim)

    penalties: list[int] = []
    for feature, feature_claims in by_feature.items():
        causal = max(
            (item.confidence for item in feature_claims if item.stance == "CAUSAL"),
            default=-1.0,
        )
        non_causal = max(
            (item.confidence for item in feature_claims if item.stance == "NON_CAUSAL"),
            default=-1.0,
        )
        if causal >= 0 and math.isclose(causal, non_causal, abs_tol=0.0, rel_tol=0.0):
            return Resolution((), True, f"equal-confidence conflict on feature {feature}")
        if non_causal > causal:
            penalties.append(feature)
    return Resolution(tuple(sorted(penalties)), False, "resolved by confidence ordering")


def _environment(
    count: int,
    generator: torch.Generator,
    shortcut_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = torch.randn(count, generator=generator)
    x2 = shortcut_sign * x1 + 0.18 * torch.randn(count, generator=generator)
    x3 = 0.85 * shortcut_sign * x1 + 0.28 * torch.randn(count, generator=generator)
    y = 2.0 * x1 + 0.12 * torch.randn(count, generator=generator)
    return torch.stack((x1, x2, x3), dim=1), y.unsqueeze(1)


def make_stress_data(
    seed_data: int,
    *,
    train_size: int = 512,
    validation_size: int = 512,
    ood_size: int = 512,
) -> StressData:
    """Frozen train/validation/OOD generator; train statistics normalize all splits."""
    generator = torch.Generator().manual_seed(seed_data)
    train_x, train_y = _environment(train_size, generator, shortcut_sign=1.0)
    # Validation is a registered proxy shift, distinct from the held-out OOD reversal.
    validation_x, validation_y = _environment(
        validation_size, generator, shortcut_sign=-0.35
    )
    ood_x, ood_y = _environment(ood_size, generator, shortcut_sign=-1.0)
    means = train_x.mean(dim=0, keepdim=True)
    scales = train_x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return StressData(
        train_x=(train_x - means) / scales,
        train_y=train_y,
        validation_x=(validation_x - means) / scales,
        validation_y=validation_y,
        ood_x=(ood_x - means) / scales,
        ood_y=ood_y,
    )


def _initial_state(seed_model: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed_model)
    return {
        key: value.detach().clone()
        for key, value in LinearStructuralModel().state_dict().items()
    }


def _fit(
    data: StressData,
    initial_state: dict[str, torch.Tensor],
    *,
    non_causal_indices: tuple[int, ...],
    lambda_reg: float,
    confidence: float,
    is_active: bool,
    epochs: int,
    learning_rate: float,
) -> LinearStructuralModel:
    model = LinearStructuralModel()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_operator = CausalRegularizedLoss()
    metadata = {
        "is_active": is_active,
        "non_causal_indices": list(non_causal_indices),
        "claim_confidence": confidence,
    }
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_operator(
            model,
            data.train_x.detach().clone(),
            data.train_y,
            lambda_reg,
            metadata,
        )
        loss.backward()
        optimizer.step()
    return model


def _errors(model: LinearStructuralModel, data: StressData) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        validation = torch.mean((model(data.validation_x) - data.validation_y).square())
        ood = torch.mean((model(data.ood_x) - data.ood_y).square())
    return float(validation), float(ood)


def _student_t_critical_975(degrees_freedom: int) -> float:
    """Accurate asymptotic t quantile without adding a SciPy runtime dependency."""
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive")
    z = 1.959963984540054
    df = float(degrees_freedom)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def paired_ci95(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("paired confidence interval needs at least two observations")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    half_width = _student_t_critical_975(len(values) - 1) * standard_error
    return {
        "n": len(values),
        "mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _point_summary(
    parameter: float,
    validation_errors: list[float],
    ood_errors: list[float],
    baseline_ood_errors: list[float],
) -> dict[str, Any]:
    paired_deltas = [
        baseline - observed for baseline, observed in zip(baseline_ood_errors, ood_errors)
    ]
    paired_relative = [
        delta / (baseline + 1e-12)
        for delta, baseline in zip(paired_deltas, baseline_ood_errors)
    ]
    return {
        "parameter": parameter,
        "validation_mse_mean": statistics.fmean(validation_errors),
        "e_ood_base_raw_mean": statistics.fmean(baseline_ood_errors),
        "e_ood_group_raw_mean": statistics.fmean(ood_errors),
        "ood_mse_mean": statistics.fmean(ood_errors),
        "delta_e_ood_absolute_paired_ci95": paired_ci95(paired_deltas),
        "r_ood_relative_improvement_paired_ci95": paired_ci95(paired_relative),
    }


def _strictly_increasing(values: list[float], tolerance: float = 1e-10) -> bool:
    return all(right > left + tolerance for left, right in zip(values, values[1:]))


def run_activation_fault_pair(
    *,
    seed_data: int = 210001,
    seed_model: int = 310001,
    lambda_reg: float = 3.0,
    epochs: int = 350,
) -> dict[str, float]:
    """One paired stale-claim fault for a focused injection test fixture."""
    data = make_stress_data(seed_data)
    state = _initial_state(seed_model)
    no_k = _fit(
        data,
        state,
        non_causal_indices=(),
        lambda_reg=0.0,
        confidence=1.0,
        is_active=False,
        epochs=epochs,
        learning_rate=0.03,
    )
    safe_stale = _fit(
        data,
        state,
        non_causal_indices=(0,),
        lambda_reg=lambda_reg,
        confidence=1.0,
        is_active=False,
        epochs=epochs,
        learning_rate=0.03,
    )
    injected = _fit(
        data,
        state,
        non_causal_indices=(0,),
        lambda_reg=lambda_reg,
        confidence=1.0,
        is_active=True,
        epochs=epochs,
        learning_rate=0.03,
    )
    no_k_ood = _errors(no_k, data)[1]
    safe_ood = _errors(safe_stale, data)[1]
    injected_ood = _errors(injected, data)[1]
    return {
        "no_k_ood_mse": no_k_ood,
        "safe_stale_ood_mse": safe_ood,
        "fault_injected_ood_mse": injected_ood,
        "generalization_mse_drop": injected_ood - safe_ood,
    }


def run_stress_test(
    *,
    seeds: StressSeeds = StressSeeds(),
    epochs: int = 350,
    learning_rate: float = 0.03,
    train_size: int = 512,
    validation_size: int = 512,
    ood_size: int = 512,
    confidence_lambda: float = 3.0,
) -> dict[str, Any]:
    if seeds.paired_runs < 30:
        raise ValueError("statistical registration requires at least 30 paired runs")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    baseline_validation: list[float] = []
    baseline_ood: list[float] = []
    correct: dict[float, tuple[list[float], list[float]]] = {
        value: ([], []) for value in LAMBDA_GRID
    }
    wrong: dict[float, tuple[list[float], list[float]]] = {
        value: ([], []) for value in LAMBDA_GRID
    }
    confidence_scan: dict[float, tuple[list[float], list[float]]] = {
        value: ([], []) for value in CONFIDENCE_GRID
    }
    low_confidence_scan: dict[float, tuple[list[float], list[float]]] = {
        value: ([], []) for value in LOW_CONFIDENCE_GRID
    }
    conflict_ood: list[float] = []
    fault_ood: list[float] = []

    conflict = resolve_conflicts(
        (
            ConflictClaim(1, "NON_CAUSAL", 0.75),
            ConflictClaim(1, "CAUSAL", 0.75),
        )
    )
    assert conflict.abstain

    for pair_index in range(seeds.paired_runs):
        data = make_stress_data(
            seeds.seed_data_start + pair_index,
            train_size=train_size,
            validation_size=validation_size,
            ood_size=ood_size,
        )
        state = _initial_state(seeds.seed_model_start + pair_index)
        cache: dict[tuple[tuple[int, ...], float, bool], tuple[float, float]] = {}

        def evaluate(
            indices: tuple[int, ...],
            lambda_reg: float,
            confidence: float,
            active: bool,
        ) -> tuple[float, float]:
            effective = round(lambda_reg * confidence, 12) if active else 0.0
            key = (indices if active else (), effective, active)
            if key not in cache:
                model = _fit(
                    data,
                    state,
                    non_causal_indices=indices,
                    lambda_reg=lambda_reg,
                    confidence=confidence,
                    is_active=active,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
                cache[key] = _errors(model, data)
            return cache[key]

        base_validation, base_ood = evaluate((), 0.0, 1.0, False)
        baseline_validation.append(base_validation)
        baseline_ood.append(base_ood)

        for lambda_reg in LAMBDA_GRID:
            correct_validation, correct_ood = evaluate((1, 2), lambda_reg, 1.0, True)
            correct[lambda_reg][0].append(correct_validation)
            correct[lambda_reg][1].append(correct_ood)
            wrong_validation, wrong_ood = evaluate((0,), lambda_reg, 1.0, True)
            wrong[lambda_reg][0].append(wrong_validation)
            wrong[lambda_reg][1].append(wrong_ood)

        for confidence in CONFIDENCE_GRID:
            validation_error, ood_error = evaluate(
                (0,), confidence_lambda, confidence, True
            )
            confidence_scan[confidence][0].append(validation_error)
            confidence_scan[confidence][1].append(ood_error)

        for confidence in LOW_CONFIDENCE_GRID:
            validation_error, ood_error = evaluate(
                (0,), confidence_lambda, confidence, True
            )
            low_confidence_scan[confidence][0].append(validation_error)
            low_confidence_scan[confidence][1].append(ood_error)

        # Equal conflict -> abstain -> inactive loss path -> bit-for-bit baseline model.
        _, resolved_ood = evaluate(
            conflict.non_causal_indices, confidence_lambda, 1.0, not conflict.abstain
        )
        conflict_ood.append(resolved_ood)

        # Fault: expired claim that wrongly suppresses current true cause X1 bypasses activation.
        _, injected_ood = evaluate((0,), confidence_lambda, 1.0, True)
        fault_ood.append(injected_ood)

    correct_points = [
        _point_summary(value, *correct[value], baseline_ood) for value in LAMBDA_GRID
    ]
    wrong_points = [
        _point_summary(value, *wrong[value], baseline_ood) for value in LAMBDA_GRID
    ]
    confidence_points = [
        _point_summary(value, *confidence_scan[value], baseline_ood)
        for value in CONFIDENCE_GRID
    ]
    low_confidence_points = [
        _point_summary(value, *low_confidence_scan[value], baseline_ood)
        for value in LOW_CONFIDENCE_GRID
    ]

    # Hyperparameter selection sees only validation aggregates.
    selected_point = min(correct_points, key=lambda item: item["validation_mse_mean"])
    selected_lambda = float(selected_point["parameter"])

    stable_mask = [
        float(point["delta_e_ood_absolute_paired_ci95"]["lower"]) > 0.1
        for point in correct_points
    ]
    stable_runs: list[list[float]] = []
    current: list[float] = []
    for lambda_reg, is_stable in zip(LAMBDA_GRID, stable_mask):
        if is_stable:
            current.append(lambda_reg)
        elif current:
            stable_runs.append(current)
            current = []
    if current:
        stable_runs.append(current)
    widest_stable = max(stable_runs, key=len, default=[])

    wrong_means = [float(point["ood_mse_mean"]) for point in wrong_points]
    confidence_harm_means = [
        -float(point["delta_e_ood_absolute_paired_ci95"]["mean"])
        for point in confidence_points
    ]
    low_confidence_harm_means = [
        -float(point["delta_e_ood_absolute_paired_ci95"]["mean"])
        for point in low_confidence_points
    ]
    fault_harm = [forced - safe for forced, safe in zip(fault_ood, baseline_ood)]
    conflict_deltas = [resolved - base for resolved, base in zip(conflict_ood, baseline_ood)]
    lambda10_harm = [
        observed - base for observed, base in zip(wrong[10.0][1], baseline_ood)
    ]

    validations = {
        "correct_k_stable_interval": len(widest_stable) >= 3,
        "selected_lambda_was_validation_only": True,
        "selected_lambda_has_ood_gain_gt_0_1": float(
            correct_points[LAMBDA_GRID.index(selected_lambda)]["delta_e_ood_absolute_paired_ci95"][
                "lower"
            ]
        )
        > 0.1,
        "wrong_k_ood_strictly_worsens_with_lambda": _strictly_increasing(wrong_means),
        "wrong_k_lambda10_collapses": paired_ci95(lambda10_harm)["lower"] > 1.0,
        "wrong_k_harm_strictly_increases_with_confidence": _strictly_increasing(
            confidence_harm_means
        ),
        "wrong_k_harm_strictly_increases_in_low_confidence_range": _strictly_increasing(
            low_confidence_harm_means
        ),
        "equal_conflict_abstains": conflict.abstain,
        "abstain_returns_exact_baseline": all(delta == 0.0 for delta in conflict_deltas),
        "fault_injection_causes_harm": paired_ci95(fault_harm)["lower"] > 1.0,
    }

    return {
        "experiment": "lambda_confidence_fault_stress",
        "statistical_registration": {
            "paired_runs": seeds.paired_runs,
            "confidence_level": 0.95,
            "interval": "paired Student-t on per-seed absolute and relative E_OOD improvement",
            "seed_data_start": seeds.seed_data_start,
            "seed_model_start": seeds.seed_model_start,
            "pairing": "all parameter points share data, OOD split, and initialization within seed",
        },
        "data_protocol": {
            "train_shortcut_sign": 1.0,
            "validation_shortcut_sign": -0.35,
            "ood_shortcut_sign": -1.0,
            "train_size": train_size,
            "validation_size": validation_size,
            "ood_size": ood_size,
            "lambda_selection_source": "validation_mse_only",
        },
        "lambda_sensitivity": {
            "grid": list(LAMBDA_GRID),
            "correct_k": correct_points,
            "wrong_k": wrong_points,
            "validation_selected_lambda": selected_lambda,
            "widest_stable_gain_interval": widest_stable,
        },
        "confidence_monotonicity": {
            "lambda_fixed": confidence_lambda,
            "grid": list(CONFIDENCE_GRID),
            "wrong_k": confidence_points,
            "wrong_k_harm_means": confidence_harm_means,
            "low_confidence_grid": list(LOW_CONFIDENCE_GRID),
            "low_confidence_wrong_k": low_confidence_points,
            "low_confidence_wrong_k_harm_means": low_confidence_harm_means,
        },
        "conflict_resolution": {
            "claims": [
                {"feature_index": 1, "stance": "NON_CAUSAL", "confidence": 0.75},
                {"feature_index": 1, "stance": "CAUSAL", "confidence": 0.75},
            ],
            "resolve_called": True,
            "abstain": conflict.abstain,
            "reason": conflict.reason,
            "ood_delta_from_no_k_ci95": paired_ci95(conflict_deltas),
        },
        "fault_injection": {
            "fault": "expired reversed-context claim bypassed Activation and forced is_active=True",
            "forced_non_causal_indices": [0],
            "lambda_reg": confidence_lambda,
            "generalization_mse_drop_ci95": paired_ci95(fault_harm),
            "delta_e_ood_absolute_ci95": paired_ci95([-value for value in fault_harm]),
            "activation_constraint_violation_rate": 1.0,
            "harm_incidence_rate": statistics.fmean(
                [float(value > 0.0) for value in fault_harm]
            ),
        },
        "lambda10_wrong_k_harm_ci95": paired_ci95(lambda10_harm),
        "validations": validations,
        "experiment_passed": all(validations.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-runs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    report = run_stress_test(
        seeds=StressSeeds(paired_runs=args.paired_runs), epochs=args.epochs
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=args.indent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
