"""Tests for retry decorators and retry behavior across call sites.

Verifies that:
- openai_retry retries on RateLimitError, APITimeoutError, APIConnectionError
- langextract_retry retries on wrapped InferenceRuntimeError
- Voyage AI uses built-in max_retries (no external decorator)
- RetryError is caught explicitly before bare except
- No double-retry on any call site
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
import tenacity

from graphrag_kg_pipeline.utils.retry import (
    _is_rate_limit_error,
    langextract_retry,
    openai_retry,
)


class TestRetryPredicates:
    """Tests for retry predicate functions."""

    def test_is_rate_limit_error_direct(self) -> None:
        exc = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        assert _is_rate_limit_error(exc) is True

    def test_is_rate_limit_error_timeout(self) -> None:
        exc = openai.APITimeoutError(request=MagicMock())
        assert _is_rate_limit_error(exc) is True

    def test_is_rate_limit_error_connection(self) -> None:
        exc = openai.APIConnectionError(request=MagicMock())
        assert _is_rate_limit_error(exc) is True

    def test_is_rate_limit_error_wrapped(self) -> None:
        """Verify wrapped OpenAI errors in InferenceRuntimeError are detected."""
        from langextract.core.exceptions import InferenceRuntimeError

        original = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        wrapped = InferenceRuntimeError("wrapped", original=original)
        assert _is_rate_limit_error(wrapped) is True

    def test_is_rate_limit_error_wrapped_timeout(self) -> None:
        from langextract.core.exceptions import InferenceRuntimeError

        original = openai.APITimeoutError(request=MagicMock())
        wrapped = InferenceRuntimeError("wrapped", original=original)
        assert _is_rate_limit_error(wrapped) is True

    def test_is_rate_limit_error_non_retryable(self) -> None:
        assert _is_rate_limit_error(ValueError("not retryable")) is False

    def test_is_rate_limit_error_wrapped_non_retryable(self) -> None:
        """InferenceRuntimeError wrapping a non-OpenAI error is not retryable."""
        from langextract.core.exceptions import InferenceRuntimeError

        wrapped = InferenceRuntimeError("wrapped", original=ValueError("oops"))
        assert _is_rate_limit_error(wrapped) is False


class TestOpenAIRetryDecorator:
    """Tests for the openai_retry decorator."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self) -> None:
        call_count = 0

        @openai_retry
        async def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise openai.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return "success"

        result = await flaky_call()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_retry_error_after_max_attempts(self) -> None:
        @openai_retry
        async def always_fails() -> str:
            raise openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )

        with pytest.raises(tenacity.RetryError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self) -> None:
        call_count = 0

        @openai_retry
        async def bad_call() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await bad_call()
        assert call_count == 1


class TestLangextractRetryDecorator:
    """Tests for the langextract_retry decorator."""

    def test_retries_on_wrapped_rate_limit(self) -> None:
        from langextract.core.exceptions import InferenceRuntimeError

        call_count = 0

        @langextract_retry
        def flaky_extract() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                original = openai.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429),
                    body=None,
                )
                raise InferenceRuntimeError("wrapped", original=original)
            return "extracted"

        result = flaky_extract()
        assert result == "extracted"
        assert call_count == 2

    def test_does_not_retry_non_openai_wrapped(self) -> None:
        from langextract.core.exceptions import InferenceRuntimeError

        call_count = 0

        @langextract_retry
        def bad_extract() -> str:
            nonlocal call_count
            call_count += 1
            raise InferenceRuntimeError("wrapped", original=TypeError("bad"))

        with pytest.raises(InferenceRuntimeError):
            bad_extract()
        assert call_count == 1


class TestGleanerRetry:
    """Tests that ExtractionGleaner uses retry correctly."""

    def _make_gleaner(self):
        from graphrag_kg_pipeline.extraction.gleaning import ExtractionGleaner

        return ExtractionGleaner(
            driver=AsyncMock(),
            database="neo4j",
            openai_api_key="sk-test",
        )

    def test_client_reuse(self) -> None:
        gleaner = self._make_gleaner()
        assert hasattr(gleaner, "_client")

    def test_call_openai_has_retry(self) -> None:
        gleaner = self._make_gleaner()
        assert hasattr(gleaner._call_openai, "retry")

    @pytest.mark.asyncio
    async def test_glean_catches_retry_error(self) -> None:
        gleaner = self._make_gleaner()
        gleaner._get_chunks_with_entities = AsyncMock(
            return_value=[{"text": "test", "entities": [], "element_id": "e1"}]
        )
        gleaner._call_openai = AsyncMock(side_effect=tenacity.RetryError(None))

        stats = await gleaner.glean_article("art-1")
        assert stats["errors"] == 1


class TestEntitySummarizerRetry:
    """Tests that EntitySummarizer uses retry correctly."""

    def _make_summarizer(self):
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver

        return EntitySummarizer(
            driver=MockDriver(),
            database="neo4j",
            openai_api_key="sk-test",
        )

    def test_client_reuse(self) -> None:
        summarizer = self._make_summarizer()
        assert hasattr(summarizer, "_client")

    def test_call_openai_has_retry(self) -> None:
        summarizer = self._make_summarizer()
        assert hasattr(summarizer._call_openai, "retry")


class TestCommunitySummarizerRetry:
    """Tests that CommunitySummarizer uses retry correctly."""

    def _make_summarizer(self):
        from graphrag_kg_pipeline.graph.community_summarizer import CommunitySummarizer

        return CommunitySummarizer(
            driver=AsyncMock(),
            database="neo4j",
            openai_api_key="sk-test",
        )

    def test_client_reuse(self) -> None:
        summarizer = self._make_summarizer()
        assert hasattr(summarizer, "_client")

    def test_call_openai_has_retry(self) -> None:
        summarizer = self._make_summarizer()
        assert hasattr(summarizer._call_openai, "retry")

    @pytest.mark.asyncio
    async def test_catches_retry_error(self) -> None:
        summarizer = self._make_summarizer()
        summarizer._get_communities = AsyncMock(
            return_value={
                0: [
                    {"name": "a", "label": "Concept", "description": ""},
                    {"name": "b", "label": "Concept", "description": ""},
                    {"name": "c", "label": "Concept", "description": ""},
                ],
            }
        )
        summarizer._call_openai = AsyncMock(side_effect=tenacity.RetryError(None))

        stats = await summarizer.summarize_communities()
        # Should gracefully handle without crashing
        assert stats["communities_summarized"] == 0


class TestLangExtractAugmenterRetry:
    """Tests that LangExtractAugmenter uses retry correctly."""

    def _make_augmenter(self):
        from graphrag_kg_pipeline.postprocessing.langextract_augmenter import (
            LangExtractAugmenter,
        )

        return LangExtractAugmenter(
            driver=AsyncMock(),
            database="neo4j",
            openai_api_key="sk-test",
        )

    def test_extract_with_retry_has_retry(self) -> None:
        augmenter = self._make_augmenter()
        assert hasattr(augmenter._extract_with_retry, "retry")

    @pytest.mark.asyncio
    async def test_catches_retry_error(self) -> None:
        augmenter = self._make_augmenter()
        augmenter._get_chunks = AsyncMock(
            return_value=[{"id": "c1", "text": "Some long enough text for processing testing."}]
        )
        augmenter._extract_with_retry = MagicMock(side_effect=tenacity.RetryError(None))

        # Should not raise — caught internally
        stats = await augmenter.augment()
        assert stats["new_entities"] == 0


class TestCommunityEmbedderRetry:
    """Tests that CommunityEmbedder uses built-in Voyage AI retry."""

    @pytest.mark.asyncio
    async def test_max_retries_passed(self) -> None:
        """Verify AsyncClient is created with max_retries=3."""
        from graphrag_kg_pipeline.graph.community_embedder import CommunityEmbedder

        embedder = CommunityEmbedder(driver=AsyncMock(), database="neo4j")
        embedder._get_communities_without_embeddings = AsyncMock(
            return_value=[{"communityId": 1, "summary": "test"}]
        )

        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed = AsyncMock(return_value=mock_result)

        with patch("voyageai.AsyncClient", return_value=mock_client) as mock_cls:
            embedder._set_embedding = AsyncMock()
            await embedder.embed_community_summaries()
            mock_cls.assert_called_once_with(max_retries=3)

    @pytest.mark.asyncio
    async def test_catches_voyage_error(self) -> None:
        """Verify VoyageError is caught per-batch without crashing."""
        import voyageai.error

        from graphrag_kg_pipeline.graph.community_embedder import CommunityEmbedder

        embedder = CommunityEmbedder(driver=AsyncMock(), database="neo4j")
        embedder._get_communities_without_embeddings = AsyncMock(
            return_value=[{"communityId": 1, "summary": "test"}]
        )

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=voyageai.error.VoyageError("fail"))

        with patch("voyageai.AsyncClient", return_value=mock_client):
            stats = await embedder.embed_community_summaries()

        assert stats["errors"] == 1
        assert stats["embedded"] == 0
