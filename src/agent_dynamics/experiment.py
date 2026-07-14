"""Reproducible 1C negative-control experiment for causal regularization."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .losses import CausalRegularizedLoss


@dataclass(frozen=True)
class ExperimentSeeds:
    seed_data: int = 104729
    seed_model: int = 130363
    seed_decode: int = 155921


@dataclass(frozen=True)
class GroupConfig:
    name: str
    lambda_reg: float
    non_causal_indices: tuple[int, ...]
    is_active: bool
    claim_confidence: float = 1.0


GROUPS = (
    GroupConfig("Group 0 (No K)", 0.0, (), False),
    GroupConfig("Group 1 (Correct K)", 4.0, (1, 2), True),
    GroupConfig("Group 2 (Wrong K)", 4.0, (0,), True),
    GroupConfig("Group 3 (Partial K)", 4.0, (1,), True),
    GroupConfig("Group 4 (Stale K)", 4.0, (1, 2), False),
    # Expired context rule bypasses Activation and suppresses X1/X3 in the new regime.
    GroupConfig("Group 5 (Stale K, fault-injected active)", 4.0, (0, 2), True),
)


@dataclass
class DataBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    ood_x: torch.Tensor
    ood_y: torch.Tensor
    true_standardized_slope: float


class LinearStructuralModel(nn.Module):
    """Small transparent model: enough capacity to expose shortcut use directly."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _environment(
    count: int,
    generator: torch.Generator,
    *,
    shortcut_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = torch.randn(count, generator=generator)
    x2 = shortcut_sign * x1 + 0.18 * torch.randn(count, generator=generator)
    x3 = 0.85 * shortcut_sign * x1 + 0.28 * torch.randn(count, generator=generator)
    y = 2.0 * x1 + 0.12 * torch.randn(count, generator=generator)
    return torch.stack((x1, x2, x3), dim=1), y.unsqueeze(1)


def make_data(seed: int, train_size: int = 1024, ood_size: int = 1024) -> DataBundle:
    """Create train/OOD environments and normalize both with train statistics."""
    generator = torch.Generator().manual_seed(seed)
    train_x, train_y = _environment(train_size, generator, shortcut_sign=1.0)
    ood_x, ood_y = _environment(ood_size, generator, shortcut_sign=-1.0)

    means = train_x.mean(dim=0, keepdim=True)
    scales = train_x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return DataBundle(
        train_x=(train_x - means) / scales,
        train_y=train_y,
        ood_x=(ood_x - means) / scales,
        ood_y=ood_y,
        true_standardized_slope=float(2.0 * scales[0, 0]),
    )


def _initial_state(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {key: value.detach().clone() for key, value in LinearStructuralModel().state_dict().items()}


def _train_group(
    config: GroupConfig,
    data: DataBundle,
    initial_state: dict[str, torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[LinearStructuralModel, float]:
    model = LinearStructuralModel()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_operator = CausalRegularizedLoss()
    claim_meta = {
        "is_active": config.is_active,
        "non_causal_indices": list(config.non_causal_indices),
        "claim_confidence": config.claim_confidence,
    }

    started = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        # Detaching each step prevents input-gradient history from leaking between steps.
        batch_x = data.train_x.detach().clone()
        loss = loss_operator(model, batch_x, data.train_y, config.lambda_reg, claim_meta)
        loss.backward()
        optimizer.step()
    return model, time.perf_counter() - started


def _metrics(
    model: LinearStructuralModel,
    data: DataBundle,
    *,
    decode_seed: int,
) -> dict[str, float | list[float]]:
    model.eval()
    with torch.no_grad():
        predictions = model(data.ood_x)
        ood_mse = torch.mean((predictions - data.ood_y).square()).item()

        # Shortcut intervention: keep X1 fixed but independently permute X2/X3.
        generator = torch.Generator().manual_seed(decode_seed)
        intervened = data.ood_x.clone()
        intervened[:, 1] = intervened[torch.randperm(len(intervened), generator=generator), 1]
        intervened[:, 2] = intervened[torch.randperm(len(intervened), generator=generator), 2]
        invariance_error = torch.mean((predictions - model(intervened)).square()).item()

        weights = model.linear.weight[0]
        sensitivity_error = abs(float(weights[0]) - data.true_standardized_slope)
        return {
            "ood_mse": ood_mse,
            "invariance_error": invariance_error,
            "sensitivity_error": sensitivity_error,
            "feature_gradients": [float(value) for value in weights],
        }


def run_experiment(
    *,
    seeds: ExperimentSeeds = ExperimentSeeds(),
    epochs: int = 350,
    learning_rate: float = 0.03,
    train_size: int = 1024,
    ood_size: int = 1024,
    timing_repeats: int = 3,
) -> dict[str, Any]:
    """Run all five groups with common data and exactly shared initialization."""
    if epochs <= 0 or train_size <= 0 or ood_size <= 0 or timing_repeats <= 0:
        raise ValueError("epochs, dataset sizes, and timing_repeats must be positive")

    # Deterministic algorithms make identical No-K/Stale-K trajectories auditable.
    torch.use_deterministic_algorithms(True)
    data = make_data(seeds.seed_data, train_size=train_size, ood_size=ood_size)
    state = _initial_state(seeds.seed_model)

    # Exclude one-time optimizer/autograd dispatch initialization from C_compute.
    _train_group(
        GROUPS[0], data, state, epochs=min(5, epochs), learning_rate=learning_rate
    )

    raw_results: list[dict[str, Any]] = []
    for group in GROUPS:
        trials = [
            _train_group(
                group,
                data,
                state,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            for _ in range(timing_repeats)
        ]
        model = trials[-1][0]
        seconds = statistics.median(trial[1] for trial in trials)
        raw_results.append(
            {
                "group": group.name,
                "lambda_reg": group.lambda_reg,
                "claim_meta": {
                    "non_causal_indices": list(group.non_causal_indices),
                    "is_active": group.is_active,
                },
                "wall_seconds": seconds,
                **_metrics(model, data, decode_seed=seeds.seed_decode),
            }
        )

    # E_OOD is the held-out shortcut-intervention error (formerly E_inv). Keep
    # predictive OOD MSE as a separate raw observation; never mix their units.
    baseline_e_ood = float(raw_results[0]["invariance_error"])
    baseline_seconds = float(raw_results[0]["wall_seconds"])
    epsilon = 1e-12
    for result in raw_results:
        group_e_ood = float(result["invariance_error"])
        delta_absolute = baseline_e_ood - group_e_ood
        relative_improvement = delta_absolute / (baseline_e_ood + epsilon)
        train_ratio = float(result["wall_seconds"]) / max(baseline_seconds, epsilon)
        result["evaluation_vector"] = {
            "e_ood_base_raw": baseline_e_ood,
            "e_ood_group_raw": group_e_ood,
            "delta_e_ood_absolute": delta_absolute,
            "r_ood_relative_improvement": relative_improvement,
            "e_inv": result["invariance_error"],
            "e_sens": result["sensitivity_error"],
            "c_train_ratio": train_ratio,
            # Every group uses the same LinearStructuralModel and one forward pass.
            "c_infer_ratio": 1.0,
        }

    correct_vector = raw_results[1]["evaluation_vector"]
    return {
        "spec_version": "v0.2.6-exp.1-final",
        "seeds": asdict(seeds),
        "timing_repeats": timing_repeats,
        "metric_definitions": {
            "e_ood_raw": "held-out shortcut-intervention MSE with X1 fixed; same raw observation as e_inv",
            "delta_e_ood_absolute": "e_ood_base_raw - e_ood_group_raw; unbounded below",
            "r_ood_relative_improvement": "delta_e_ood_absolute / (e_ood_base_raw + 1e-12); upper-bounded by 1",
            "predictive_ood_mse_raw": "held-out label prediction MSE; retained separately and never used in delta_e_ood",
            "e_inv": "alias of e_ood_group_raw; lower is better",
            "e_sens": "absolute error of dŷ/dX1 from the structural slope; lower is better",
            "c_train_ratio": "median group training wall time divided by No-K wall time",
            "c_infer_ratio": "forward-pass graph cost divided by No-K; identical architecture gives 1",
        },
        "headline_correct_k": {
            "e_ood_base_raw": correct_vector["e_ood_base_raw"],
            "e_ood_group_correct_k_raw": correct_vector["e_ood_group_raw"],
            "delta_e_ood_absolute": correct_vector["delta_e_ood_absolute"],
            "r_ood_relative_improvement": correct_vector["r_ood_relative_improvement"],
            "c_train_ratio": correct_vector["c_train_ratio"],
            "c_infer_ratio": correct_vector["c_infer_ratio"],
        },
        "results": raw_results,
    }


def evaluate_ir_execution(
    claim: dict[str, Any],
    *,
    seeds: ExperimentSeeds = ExperimentSeeds(),
    epochs: int = 120,
    learning_rate: float = 0.03,
    data_size: int = 512,
) -> dict[str, float]:
    """Execute an IR through the 1C loss path and return its OOD error vector.

    Only an admitted ``VALIDATED`` claim may activate regularization. Candidate
    compiler output therefore remains safe until a separate admission transition.
    """
    constraints = claim.get("scope", {}).get("context_constraints", {})
    raw_indices = constraints.get("non_causal_indices", [])
    indices = tuple(int(index) for index in raw_indices)
    active = claim.get("status") == "VALIDATED" and bool(constraints.get("is_active", True))
    config = GroupConfig(
        "IR execution",
        float(constraints.get("lambda_reg", 4.0)),
        indices,
        active,
        float(claim.get("claim_confidence") or 0.0),
    )
    data = make_data(seeds.seed_data, train_size=data_size, ood_size=data_size)
    model, _ = _train_group(
        config,
        data,
        _initial_state(seeds.seed_model),
        epochs=epochs,
        learning_rate=learning_rate,
    )
    metrics = _metrics(model, data, decode_seed=seeds.seed_decode)
    return {
        "e_inv": float(metrics["invariance_error"]),
        "e_sens": float(metrics["sensitivity_error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--ood-size", type=int, default=1024)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        epochs=args.epochs,
        train_size=args.train_size,
        ood_size=args.ood_size,
        timing_repeats=args.timing_repeats,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=args.indent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
