from __future__ import annotations

import pytest
import torch
from torch import nn

from agent_dynamics.losses import CausalRegularizedLoss


def _fixed_model() -> nn.Linear:
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0, 4.0]]))
    return model


def test_active_claim_adds_confidence_weighted_shortcut_gradient_penalty() -> None:
    x = torch.ones((5, 3))
    y = _fixed_model()(x).detach()
    loss = CausalRegularizedLoss()(
        _fixed_model(),
        x,
        y,
        lambda_reg=2.0,
        claim_meta={
            "is_active": True,
            "non_causal_indices": [1, 2],
            "claim_confidence": 0.5,
        },
    )
    # Task loss is zero; gradients are 3 and 4, so 2 * .5 * (3² + 4²) = 25.
    assert loss.item() == pytest.approx(25.0)


def test_inactive_claim_is_exactly_task_loss() -> None:
    model = _fixed_model()
    x = torch.ones((2, 3))
    y = torch.zeros((2, 1))
    operator = CausalRegularizedLoss()
    actual = operator(
        model,
        x,
        y,
        lambda_reg=100.0,
        claim_meta={"is_active": False, "non_causal_indices": [0, 1, 2]},
    )
    expected = nn.MSELoss()(model(x), y)
    assert torch.equal(actual, expected)


def test_unscored_candidate_is_safe_when_inactive_and_rejected_when_active() -> None:
    model = _fixed_model()
    x = torch.ones((2, 3))
    y = torch.zeros((2, 1))
    operator = CausalRegularizedLoss()
    inactive = operator(
        model,
        x,
        y,
        1.0,
        {"is_active": False, "claim_confidence": None, "non_causal_indices": [1]},
    )
    assert torch.equal(inactive, nn.MSELoss()(model(x), y))
    with pytest.raises(ValueError, match="epistemic data plane"):
        operator(
            model,
            x.detach(),
            y,
            1.0,
            {"is_active": True, "claim_confidence": None, "non_causal_indices": [1]},
        )


@pytest.mark.parametrize("indices,error", [([3], IndexError), ([1.0], TypeError)])
def test_invalid_feature_index_is_rejected(indices: list[object], error: type[Exception]) -> None:
    with pytest.raises(error):
        CausalRegularizedLoss()(
            _fixed_model(),
            torch.ones((2, 3)),
            torch.zeros((2, 1)),
            1.0,
            {"is_active": True, "non_causal_indices": indices},
        )
