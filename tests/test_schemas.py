from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, FormatChecker


ROOT = Path(__file__).parents[1]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["audit_event.schema.json", "causal_claim.schema.json"])
def test_schema_is_valid_draft_07(name: str) -> None:
    Draft7Validator.check_schema(_schema(name))


def test_audit_event_rejects_non_sha256_hash() -> None:
    instance = {
        "event_id": "7b83557f-3983-4f7e-bf1b-fc9a26781f10",
        "actor": "feed",
        "timestamp": "2026-07-13T00:00:00Z",
        "source": "api",
        "payload_hash": "not-a-hash",
        "size_bytes": 1,
        "parse_status": "PENDING",
    }
    errors = list(
        Draft7Validator(_schema("audit_event.schema.json"), format_checker=FormatChecker())
        .iter_errors(instance)
    )
    assert any(error.validator == "pattern" for error in errors)


def test_causal_claim_accepts_frozen_contract() -> None:
    instance = {
        "id": "7b83557f-3983-4f7e-bf1b-fc9a26781f10",
        "cause": "X1",
        "effect": "Y",
        "scope": {"market_phase": "risk-on", "context_constraints": {}},
        "time_window": {
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": "2026-12-31T00:00:00Z",
            "lag_order": 0,
        },
        "effect_spec": {"estimand": "CATE", "estimate": [2.0], "uncertainty": 0.1},
        "claim_confidence": 0.95,
        "provenance": "experiment:1C",
        "version": 1,
        "status": "VALIDATED",
    }
    Draft7Validator(
        _schema("causal_claim.schema.json"), format_checker=FormatChecker()
    ).validate(instance)


def test_unvalidated_candidate_must_support_null_confidence() -> None:
    instance = {
        "id": "7b83557f-3983-4f7e-bf1b-fc9a26781f10",
        "cause": "X1",
        "effect": "Y",
        "scope": {"market_phase": "UNSPECIFIED"},
        "time_window": {
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": "2026-12-31T00:00:00Z"
        },
        "effect_spec": {"estimand": "ATE", "estimate": [], "uncertainty": 0.0},
        "claim_confidence": None,
        "provenance": "user_supplied:natural_language",
        "version": 1,
        "status": "CANDIDATE",
    }
    Draft7Validator(
        _schema("causal_claim.schema.json"), format_checker=FormatChecker()
    ).validate(instance)
