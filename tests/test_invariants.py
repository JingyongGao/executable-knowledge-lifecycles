from __future__ import annotations

import pytest

from agent_dynamics.system import (
    AuthError,
    EnforcementPlane,
    EventAuditingPlane,
    EpistemicDataPlane,
    ProtectedEpistemicStore,
)


@pytest.fixture
def protected_store() -> ProtectedEpistemicStore:
    store = ProtectedEpistemicStore()
    store.append_belief(0, {"state": "trusted"}, actor="EpistemicUpdater")
    return store


@pytest.fixture
def enforcement_plane() -> EnforcementPlane:
    return EnforcementPlane()


def test_system_core_invariants(
    enforcement_plane: EnforcementPlane,
    protected_store: ProtectedEpistemicStore,
) -> None:
    # 1. Historical B_<t snapshots are physically denied on every overwrite path.
    with pytest.raises(PermissionError):
        protected_store.write_historical_belief(
            time_step=0,
            data={"state": "corrupted"},
            actor="untrusted-principal",
        )

    # 2. Normative/action/value layers cannot write down into the evidence ledger.
    with pytest.raises(AuthError):
        enforcement_plane.route_request(
            principal="NormativePlanner",
            action="MUTATE_EVIDENCE",
            resource="ProtectedEpistemicStore/Provenance",
        )


def test_belief_reads_are_detached_from_store(protected_store: ProtectedEpistemicStore) -> None:
    read_copy = protected_store.read_belief(0)
    read_copy["state"] = "locally-corrupted"
    assert protected_store.read_belief(0) == {"state": "trusted"}


def test_normative_principal_may_read_but_not_mutate_evidence(
    enforcement_plane: EnforcementPlane,
) -> None:
    assert enforcement_plane.route_request(
        principal="NormativePlanner",
        action="READ_EVIDENCE",
        resource="ProtectedEpistemicStore/Provenance",
    )


def test_external_event_is_audited_even_when_parsing_fails() -> None:
    audit = EventAuditingPlane()

    def reject(_: bytes) -> None:
        raise ValueError("malformed")

    with pytest.raises(ValueError, match="malformed"):
        audit.ingest(b"bad input", actor="source-A", source="api", parser=reject)

    assert len(audit.ledger) == 1
    assert audit.ledger[0]["parse_status"] == "PARSE_FAILED"
    assert audit.ledger[0]["size_bytes"] == len(b"bad input")


def test_only_epistemic_plane_assigns_candidate_confidence() -> None:
    candidate = {"status": "CANDIDATE", "claim_confidence": None}
    validated = EpistemicDataPlane().validate_candidate(
        candidate,
        provenance_records=5,
        sample_size=500,
        consistency_score=0.9,
    )
    assert validated["status"] == "VALIDATED"
    assert 0.0 < validated["claim_confidence"] < 1.0
    assert candidate["claim_confidence"] is None


def test_epistemic_plane_rejects_pre_scored_candidate() -> None:
    with pytest.raises(AuthError):
        EpistemicDataPlane().validate_candidate(
            {"status": "CANDIDATE", "claim_confidence": 1.0},
            provenance_records=5,
            sample_size=500,
            consistency_score=1.0,
        )
