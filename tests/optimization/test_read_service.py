from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.optimization.experience import EvidenceClassification
from app.optimization.memory import StrategyCandidateRecord
from app.optimization.read_service import read_candidates
from app.optimization.schemas import CandidateQuery
from tests.optimization.test_experience import profile


def candidate(candidate_id: str) -> StrategyCandidateRecord:
    return StrategyCandidateRecord(
        stable_candidate_id=candidate_id,
        score_model_name="baseline_technical",
        score_model_version=1,
        strategy_name="baseline_long_only",
        strategy_version=1,
        score_parameters={
            "trend_weight": "0.40",
            "rsi_weight": "0.30",
            "return_1_weight": "0.15",
            "return_12_weight": "0.15",
            "volume_confirmation_weight": "0.25",
        },
        strategy_parameters={
            "entry_threshold": "0.60",
            "exit_threshold": "0.20",
            "open_position_target_percentage": "15",
        },
        canonical_parameters={"ordered_values": []},
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_candidate_keyset_scans_beyond_non_matching_batches_without_loss() -> (
    None
):
    definitions = {value: candidate(value) for value in ("a", "b", "c", "d", "e")}
    profiles = {
        value: replace(
            profile(stable_candidate_id=value),
            evidence_classification=(
                EvidenceClassification.PROMISING
                if value in {"c", "d", "e"}
                else EvidenceClassification.INSUFFICIENT_EVIDENCE
            ),
        )
        for value in definitions
    }

    async def batch(
        _session: object,
        _query: CandidateQuery,
        *,
        after_candidate_id: str | None,
        batch_size: int,
    ) -> tuple[StrategyCandidateRecord, ...]:
        assert batch_size == 2
        ordered = [
            value
            for value in definitions
            if after_candidate_id is None or value > after_candidate_id
        ]
        return tuple(definitions[value] for value in ordered[:batch_size])

    async def evidence(
        _session: object,
        *,
        candidate_ids: tuple[str, ...],
        query: CandidateQuery,
    ) -> list[str]:
        assert query.evidence_classification is EvidenceClassification.PROMISING
        return list(candidate_ids)

    def composed(rows: list[str]) -> dict[str, object]:
        return {value: profiles[value] for value in rows}

    query = CandidateQuery(
        evidence_classification=EvidenceClassification.PROMISING, limit=2
    )
    with (
        patch("app.optimization.read_service.CANDIDATE_SCAN_BATCH_SIZE", 2),
        patch(
            "app.optimization.read_service.list_candidate_definition_batch",
            side_effect=batch,
        ),
        patch(
            "app.optimization.read_service.load_profile_evidence",
            side_effect=evidence,
        ),
        patch("app.optimization.read_service._profiles", side_effect=composed),
    ):
        first = await read_candidates(AsyncMock(), query)
        second = await read_candidates(
            AsyncMock(), query.model_copy(update={"cursor": first.next_cursor})
        )

    assert [item.definition.stable_candidate_id for item in first.items] == ["c", "d"]
    assert first.has_more
    assert first.next_cursor == "d"
    assert [item.definition.stable_candidate_id for item in second.items] == ["e"]
    assert not second.has_more
    assert second.next_cursor is None
