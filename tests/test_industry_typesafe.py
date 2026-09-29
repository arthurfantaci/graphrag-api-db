"""Tests for TypeSafe-backed industry classification.

Covers the client factory, the Choice criteria tables, the answer-to-action
mapping, the batched async classifier, and the normalizer's key requirement.
All ``system_one()`` calls are mocked with real SDK response models; no live
API calls run in CI.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, Usage

from graphrag_kg_pipeline.exceptions import TypeSafeConfigError
from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
    CANONICAL_INDUSTRIES,
    INDUSTRY_CRITERIA,
    NON_INDUSTRY_CRITERIA,
    IndustryJudgment,
    IndustryNormalizer,
    build_industry_choice_criteria,
    build_industry_questions,
    classify_by_table,
    classify_industry_terms,
    judgment_to_action,
)
from graphrag_kg_pipeline.utils.typesafe_client import (
    TYPESAFE_RETRY_POLICY,
    create_typesafe_client,
)


def _response(answers: dict[str, tuple[str, float]]) -> SystemOneResponse:
    """Build a real SDK response from ``{question_id: (choice, confidence)}``."""
    built: dict[str, ChoiceAnswer] = {}
    for key, (choice, confidence) in answers.items():
        remainder = round(1.0 - confidence, 4)
        built[key] = ChoiceAnswer(
            choice=choice,
            confidence=confidence,
            probabilities={choice: confidence, "none_of_these": remainder},
        )
    return SystemOneResponse(model="jev-test", usage=Usage(), answers=built)


# =============================================================================
# CLIENT FACTORY
# =============================================================================


class TestCreateTypesafeClient:
    """Tests for the client factory and its retry policy."""

    def test_empty_key_raises_config_error(self) -> None:
        with pytest.raises(TypeSafeConfigError, match="TYPESAFE_API_KEY"):
            create_typesafe_client("")

    def test_whitespace_key_raises_config_error(self) -> None:
        with pytest.raises(TypeSafeConfigError):
            create_typesafe_client("   ")

    def test_valid_key_returns_async_client_with_retry_policy(self) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        client = create_typesafe_client("ts-test-key-1234")
        assert isinstance(client, AsyncTypeSafeClient)

    def test_retry_policy_matches_project_convention(self) -> None:
        """Three retries, like openai_retry; the SDK handles backoff itself."""
        assert TYPESAFE_RETRY_POLICY.max_retries == 3


# =============================================================================
# CRITERIA TABLES
# =============================================================================


class TestCriteria:
    """Tests for the Choice criteria tables."""

    def test_every_canonical_industry_has_a_description(self) -> None:
        assert set(INDUSTRY_CRITERIA) == set(CANONICAL_INDUSTRIES)

    def test_descriptions_are_non_empty_strings(self) -> None:
        for name, description in INDUSTRY_CRITERIA.items():
            assert isinstance(description, str), name
            assert description.strip(), name

    def test_non_industry_options_are_the_four_dispositions(self) -> None:
        assert set(NON_INDUSTRY_CRITERIA) == {
            "organization",
            "concept_not_industry",
            "too_generic",
            "none_of_these",
        }

    def test_combined_criteria_have_no_overlap(self) -> None:
        combined = build_industry_choice_criteria()
        assert len(combined) == len(INDUSTRY_CRITERIA) + len(NON_INDUSTRY_CRITERIA)
        assert len(combined) <= 255  # Choice option ceiling from the API docs


# =============================================================================
# EXACT-MATCH FAST PATH
# =============================================================================


class TestClassifyByTable:
    """Tests for the exact-match fast path."""

    def test_exact_taxonomy_hit(self) -> None:
        assert classify_by_table("Auto Industry") == ("keep", "automotive")

    def test_organization_hit(self) -> None:
        assert classify_by_table("FDA") == ("reclassify_org", None)

    def test_concept_hit(self) -> None:
        assert classify_by_table("machine learning") == ("reclassify", None)

    def test_generic_hit(self) -> None:
        assert classify_by_table("regulated industries") == ("delete", None)

    def test_empty_name_is_deleted(self) -> None:
        assert classify_by_table("") == ("delete", None)

    def test_near_miss_is_not_fuzzy_matched(self) -> None:
        """The fast path is exact only; 'retail' must not become 'rail'."""
        assert classify_by_table("retail") is None


# =============================================================================
# ANSWER -> ACTION MAPPING
# =============================================================================


class TestJudgmentToAction:
    """Tests for mapping a Choice answer onto a consolidator action."""

    def test_canonical_industry_keeps(self) -> None:
        assert judgment_to_action("rail", 0.9, min_confidence=0.5) == ("keep", "rail")

    def test_organization_reclassifies_to_org(self) -> None:
        assert judgment_to_action("organization", 0.9, min_confidence=0.5) == (
            "reclassify_org",
            None,
        )

    def test_concept_reclassifies(self) -> None:
        assert judgment_to_action("concept_not_industry", 0.9, min_confidence=0.5) == (
            "reclassify",
            None,
        )

    def test_generic_deletes(self) -> None:
        assert judgment_to_action("too_generic", 0.9, min_confidence=0.5) == ("delete", None)

    def test_none_of_these_is_unknown(self) -> None:
        assert judgment_to_action("none_of_these", 0.9, min_confidence=0.5) == ("unknown", None)

    def test_low_confidence_demotes_to_unknown(self) -> None:
        assert judgment_to_action("rail", 0.3, min_confidence=0.5) == ("unknown", None)

    def test_confidence_at_floor_is_accepted(self) -> None:
        assert judgment_to_action("rail", 0.5, min_confidence=0.5) == ("keep", "rail")

    def test_unexpected_option_is_unknown(self) -> None:
        assert judgment_to_action("not_an_option", 1.0, min_confidence=0.0) == ("unknown", None)


# =============================================================================
# QUESTION BUILDER
# =============================================================================


class TestBuildIndustryQuestions:
    """Tests for the per-term Choice question builder."""

    def test_one_choice_per_term_keyed_by_index(self) -> None:
        questions = build_industry_questions(["retail", "pharmacy"])
        assert list(questions) == ["term_0", "term_1"]

    def test_instructions_reference_state_path(self) -> None:
        questions = build_industry_questions(["retail"])
        assert "`terms[0]`" in str(questions["term_0"].instructions)

    def test_criteria_are_the_full_option_set(self) -> None:
        questions = build_industry_questions(["retail"])
        assert set(questions["term_0"].criteria) == set(build_industry_choice_criteria())


# =============================================================================
# BATCHED ASYNC CLASSIFIER
# =============================================================================


class TestClassifyIndustryTerms:
    """Tests for the batched async classifier with a mocked client."""

    async def test_maps_answers_back_to_terms(self) -> None:
        client = AsyncMock()
        client.system_one.return_value = _response(
            {
                "term_0": ("none_of_these", 0.95),
                "term_1": ("life sciences", 0.6),
                "term_2": ("too_generic", 0.88),
            }
        )

        judgments = await classify_industry_terms(
            client, ["retail", "pharmacy", "various sectors"], min_confidence=0.5
        )

        assert [j.term for j in judgments] == ["retail", "pharmacy", "various sectors"]
        assert judgments[0].action == "unknown"
        assert judgments[1] == IndustryJudgment(
            term="pharmacy",
            choice="life sciences",
            confidence=0.6,
            probabilities={"life sciences": 0.6, "none_of_these": 0.4},
            action="keep",
            canonical="life sciences",
        )
        assert judgments[2].action == "delete"

    async def test_sends_terms_in_state_and_one_question_each(self) -> None:
        client = AsyncMock()
        client.system_one.return_value = _response({"term_0": ("rail", 0.9)})

        await classify_industry_terms(client, ["railway systems"])

        _, kwargs = client.system_one.call_args
        assert kwargs["state"]["terms"] == ["railway systems"]
        assert "context" in kwargs["state"]
        assert list(kwargs["questions"]) == ["term_0"]

    async def test_splits_into_batches(self) -> None:
        terms = [f"term {i}" for i in range(120)]
        client = AsyncMock()

        def _answer(*, questions: dict, **_: object) -> SystemOneResponse:
            return _response(dict.fromkeys(questions, ("none_of_these", 0.9)))

        client.system_one.side_effect = _answer

        judgments = await classify_industry_terms(client, terms, batch_size=50)

        assert client.system_one.await_count == 3
        assert [j.term for j in judgments] == terms

    async def test_low_confidence_is_demoted(self) -> None:
        client = AsyncMock()
        client.system_one.return_value = _response({"term_0": ("rail", 0.2)})

        judgments = await classify_industry_terms(client, ["retail"], min_confidence=0.5)

        assert judgments[0].choice == "rail"
        assert judgments[0].action == "unknown"

    async def test_empty_input_makes_no_call(self) -> None:
        client = AsyncMock()
        assert await classify_industry_terms(client, []) == []
        client.system_one.assert_not_awaited()


# =============================================================================
# NORMALIZER
# =============================================================================


class TestIndustryNormalizerRequiresClient:
    """Tests that the normalizer refuses to run without a client."""

    def test_missing_client_raises_config_error(self, mock_neo4j_driver: object) -> None:
        with pytest.raises(TypeSafeConfigError):
            IndustryNormalizer(mock_neo4j_driver)  # type: ignore[arg-type]


class TestIndustryNormalizerConsolidation:
    """Tests for routing table hits and model judgments through consolidation."""

    async def test_table_hits_skip_the_model(self, mock_neo4j_driver: object) -> None:
        client = AsyncMock()
        normalizer = IndustryNormalizer(mock_neo4j_driver, typesafe_client=client)  # type: ignore[arg-type]
        industries = [
            {"name": "auto industry", "display_name": None, "element_id": "e1"},
            {"name": "fda", "display_name": None, "element_id": "e2"},
        ]

        with (
            patch.object(normalizer, "get_current_industries", AsyncMock(return_value=industries)),
            patch.object(normalizer, "_update_industry_name", AsyncMock()) as rename,
            patch.object(normalizer, "_reclassify_to_organization", AsyncMock()) as to_org,
        ):
            stats = await normalizer.consolidate_industries()

        client.system_one.assert_not_awaited()
        rename.assert_awaited_once_with("e1", "automotive")
        to_org.assert_awaited_once_with("e2")
        assert stats["table_resolved"] == 2
        assert stats["model_resolved"] == 0

    async def test_unresolved_terms_go_to_the_model(self, mock_neo4j_driver: object) -> None:
        client = AsyncMock()
        client.system_one.return_value = _response(
            {"term_0": ("none_of_these", 0.97), "term_1": ("too_generic", 0.9)}
        )
        normalizer = IndustryNormalizer(mock_neo4j_driver, typesafe_client=client)  # type: ignore[arg-type]
        industries = [
            {"name": "retail", "display_name": None, "element_id": "e1"},
            {"name": "various sectors", "display_name": None, "element_id": "e2"},
        ]

        with (
            patch.object(normalizer, "get_current_industries", AsyncMock(return_value=industries)),
            patch.object(normalizer, "_delete_industry", AsyncMock()) as delete,
            patch.object(normalizer, "_update_industry_name", AsyncMock()) as rename,
        ):
            stats = await normalizer.consolidate_industries()

        _, kwargs = client.system_one.call_args
        assert kwargs["state"]["terms"] == ["retail", "various sectors"]
        delete.assert_awaited_once_with("e2")
        rename.assert_not_awaited()
        assert stats["unknown"] == ["retail"]
        assert stats["model_resolved"] == 2
        assert [j["term"] for j in stats["judgments"]] == ["retail", "various sectors"]
        assert stats["judgments"][0]["confidence"] == 0.97


class TestClassifyIndustryTermsFailure:
    """A failed batch must not abort consolidation or lose other batches."""

    async def test_failed_batch_yields_unknown_and_later_batches_continue(self) -> None:
        from typesafe_sdk import TypeSafeError

        client = AsyncMock()
        client.system_one.side_effect = [
            _response({"term_0": ("rail", 0.9), "term_1": ("rail", 0.9)}),
            TypeSafeError("503 after retries"),
            _response({"term_0": ("energy", 0.8)}),
        ]

        judgments = await classify_industry_terms(
            client, ["a", "b", "c", "d", "e"], batch_size=2, min_confidence=0.5
        )

        assert [j.action for j in judgments] == ["keep", "keep", "unknown", "unknown", "keep"]
        assert judgments[2].error == "503 after retries"
        assert judgments[2].confidence == 0.0
        assert judgments[0].error is None
        assert client.system_one.await_count == 3
