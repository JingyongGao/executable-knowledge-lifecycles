"""Minimal audit, authorization, and protected-store enforcement planes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4


class AuthError(PermissionError):
    """Raised when a principal crosses an authorization boundary."""


@dataclass(frozen=True)
class RawEventLedgerEntry:
    event_id: str
    actor: str
    timestamp: str
    source: str
    payload_hash: str
    size_bytes: int
    parse_status: str
    content_type: str | None = None


class EventAuditingPlane:
    """Append-only metadata ledger placed before every external parser."""

    def __init__(self) -> None:
        self._ledger: list[RawEventLedgerEntry] = []

    @property
    def ledger(self) -> tuple[Mapping[str, Any], ...]:
        # Return detached, read-only snapshots; callers cannot mutate ledger history.
        return tuple(MappingProxyType(asdict(entry)) for entry in self._ledger)

    def ingest(
        self,
        payload: bytes,
        *,
        actor: str,
        source: str,
        parser: Callable[[bytes], Any],
        content_type: str | None = None,
    ) -> Any:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes so its audited hash and size are unambiguous")

        position = len(self._ledger)
        self._ledger.append(
            RawEventLedgerEntry(
                event_id=str(uuid4()),
                actor=actor,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=source,
                payload_hash=sha256(payload).hexdigest(),
                size_bytes=len(payload),
                content_type=content_type,
                parse_status="PENDING",
            )
        )
        try:
            result = parser(payload)
        except Exception:
            self._replace_status(position, "PARSE_FAILED")
            raise
        self._replace_status(position, "PARSED")
        return result

    def _replace_status(self, position: int, status: str) -> None:
        old = asdict(self._ledger[position])
        old["parse_status"] = status
        self._ledger[position] = RawEventLedgerEntry(**old)


class ProtectedEpistemicStore:
    """Append-only belief snapshots and versioned causal claims."""

    def __init__(self) -> None:
        self._beliefs: dict[int, Any] = {}
        self._claims: dict[str, list[dict[str, Any]]] = {}

    def append_belief(self, time_step: int, data: Any, *, actor: str) -> None:
        if time_step in self._beliefs:
            raise PermissionError(f"belief snapshot B_{time_step} is immutable")
        self._beliefs[time_step] = deepcopy(data)

    def read_belief(self, time_step: int) -> Any:
        return deepcopy(self._beliefs[time_step])

    def write_historical_belief(
        self, time_step: int, data: Any, *, actor: str
    ) -> None:
        """Explicitly denied path retained as a testable system boundary."""
        raise PermissionError(f"historical belief B_{time_step} cannot be overwritten")

    def put_causal_claim(self, claim: Mapping[str, Any], *, actor: str) -> None:
        detached = deepcopy(dict(claim))
        claim_id = str(detached["id"])
        version = int(detached["version"])
        versions = self._claims.setdefault(claim_id, [])
        expected_version = len(versions) + 1
        if version != expected_version:
            raise ValueError(
                f"claim {claim_id} version must be {expected_version}, got {version}"
            )
        versions.append(detached)

    def claim_versions(self, claim_id: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(deepcopy(item)) for item in self._claims.get(claim_id, []))


class EpistemicDataPlane:
    """Sole authority that converts an unscored candidate into a validated claim."""

    def __init__(self, sample_scale: float = 100.0) -> None:
        if sample_scale <= 0:
            raise ValueError("sample_scale must be positive")
        self.sample_scale = sample_scale

    def validate_candidate(
        self,
        claim: Mapping[str, Any],
        *,
        provenance_records: int,
        sample_size: int,
        consistency_score: float,
    ) -> dict[str, Any]:
        if claim.get("status") != "CANDIDATE" or claim.get("claim_confidence") is not None:
            raise AuthError("validation accepts only unscored CANDIDATE claims")
        if provenance_records <= 0 or sample_size <= 0:
            raise ValueError("validation requires positive lineage and sample counts")
        if not 0.0 <= consistency_score <= 1.0:
            raise ValueError("consistency_score must be in [0, 1]")

        lineage_factor = 1.0 - math.exp(-provenance_records / 3.0)
        sample_factor = 1.0 - math.exp(-sample_size / self.sample_scale)
        confidence = max(
            0.0, min(1.0, lineage_factor * sample_factor * consistency_score)
        )
        validated = deepcopy(dict(claim))
        validated["claim_confidence"] = confidence
        validated["status"] = "VALIDATED"
        return validated


class EnforcementPlane:
    """Deny normative-to-evidence write-down while allowing explicit safe routes."""

    _EVIDENCE_MUTATIONS = {"MUTATE_EVIDENCE", "DELETE_EVIDENCE", "REWRITE_PROVENANCE"}
    _NORMATIVE_PRINCIPALS = {"NormativePlanner", "ActionProposal", "ValueOptimizer"}
    _EPISTEMIC_RESOURCES = ("ProtectedEpistemicStore/", "RawEventLedger/")

    def route_request(self, *, principal: str, action: str, resource: str) -> bool:
        is_epistemic_resource = resource.startswith(self._EPISTEMIC_RESOURCES)
        if (
            principal in self._NORMATIVE_PRINCIPALS
            and action in self._EVIDENCE_MUTATIONS
            and is_epistemic_resource
        ):
            raise AuthError(
                f"{principal} is not authorized to {action} on epistemic resource {resource}"
            )
        return True
