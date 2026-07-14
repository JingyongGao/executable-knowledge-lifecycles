"""Differentiable causal-knowledge enforcement operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class CausalRegularizedLoss(nn.Module):
    """Task loss plus an input-gradient penalty for declared shortcuts.

    This implements

        L(K) = L_task + lambda * 1[Active(K)] * c_K
               * E_x[sum(j in N(K)) ||df_theta(x)/dx_j||²].

    ``claim_meta`` is deliberately passed per call because claim activation is a
    property of the current time and scope, not a static model setting.
    """

    def __init__(self, base_criterion: nn.Module | None = None) -> None:
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss()

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        lambda_reg: float,
        claim_meta: Mapping[str, Any],
    ) -> torch.Tensor:
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")

        active = bool(claim_meta.get("is_active", False))
        raw_confidence = claim_meta.get("claim_confidence", 1.0)
        if raw_confidence is None:
            if active:
                raise ValueError("an active claim must be scored by the epistemic data plane")
            confidence = 0.0
        else:
            confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("claim_confidence must be in [0, 1]")

        # Preserve the caller's tensor identity while making differentiation possible.
        if not x.requires_grad:
            x.requires_grad_(True)

        y_pred = model(x)
        task_loss = self.base_criterion(y_pred, y_true)

        if not active or lambda_reg == 0:
            return task_loss

        indices = self._validate_indices(
            claim_meta.get("non_causal_indices", []), feature_count=x.shape[-1]
        )
        if not indices:
            return task_loss

        # For vector outputs this is the gradient of their sum, matching the frozen
        # reference operator's all-ones vector-Jacobian product.
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        reg_term = torch.stack(
            [torch.mean(gradients[..., idx].square()) for idx in indices]
        ).sum()
        return task_loss + float(lambda_reg) * confidence * reg_term

    @staticmethod
    def _validate_indices(raw_indices: object, feature_count: int) -> tuple[int, ...]:
        if isinstance(raw_indices, (str, bytes)) or not isinstance(raw_indices, Sequence):
            raise TypeError("non_causal_indices must be a sequence of integers")

        indices: list[int] = []
        for idx in raw_indices:
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise TypeError("non_causal_indices must contain only integers")
            if idx < 0 or idx >= feature_count:
                raise IndexError(f"non-causal feature index {idx} is out of bounds")
            if idx not in indices:
                indices.append(idx)
        return tuple(indices)
