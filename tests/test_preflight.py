"""Tests for pre-flight validation checks.

Verifies each check produces the correct result on success and raises
PreflightError with a helpful message on failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_kg_pipeline.preflight import (
    PreflightError,
    PreflightResult,
    _check_apoc,
    _check_existing_data,
    _check_neo4j_connectivity,
    _check_vector_index,
    _check_voyage_api,
    run_preflight_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(data: dict) -> MagicMock:
    """Create a mock Neo4j record that supports dict-style access and dict() conversion."""
    record = MagicMock()
    record.__getitem__ = lambda _self, k: data[k]
    record.keys = MagicMock(return_value=data.keys())
    return record


class _AsyncRecordIterator:
    """Async iterator over mock Neo4j records."""

    def __init__(self, records: list[MagicMock]) -> None:
        self._records = iter(records)

    def __aiter__(self) -> _AsyncRecordIterator:
        return self

    async def __anext__(self) -> MagicMock:
        try:
            return next(self._records)
        except StopIteration:
            raise StopAsyncIteration from None


def _make_async_result(records: list[dict]) -> AsyncMock:
    """Create a mock Neo4j result with async iteration and single() support."""
    mock_records = [_make_record(r) for r in records]
    mock_result = MagicMock()
    mock_result.single = AsyncMock(return_value=mock_records[0] if mock_records else None)
    mock_result.__aiter__ = lambda _self: _AsyncRecordIterator(mock_records)
    return mock_result


def _mock_driver(records: list[dict] | None = None) -> AsyncMock:
    """Create a mock async Neo4j driver that returns given records."""
    driver = AsyncMock()
    session = AsyncMock()

    if records is not None:
        session.run = AsyncMock(return_value=_make_async_result(records))
    else:
        session.run = AsyncMock(side_effect=Exception("Connection refused"))

    driver.session = MagicMock(return_value=session)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return driver


# ---------------------------------------------------------------------------
# Neo4j Connectivity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neo4j_connectivity_success() -> None:
    """Passes when Neo4j returns expected ping response."""
    driver = _mock_driver([{"ping": 1}])
    result = PreflightResult()
    await _check_neo4j_connectivity(driver, "neo4j", result)
    assert result.neo4j_connected is True


@pytest.mark.asyncio
async def test_neo4j_connectivity_failure() -> None:
    """Raises PreflightError when Neo4j is unreachable."""
    driver = _mock_driver(None)
    result = PreflightResult()
    with pytest.raises(PreflightError, match="Cannot connect to Neo4j"):
        await _check_neo4j_connectivity(driver, "neo4j", result)


# ---------------------------------------------------------------------------
# APOC Check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apoc_check_success() -> None:
    """Passes when APOC returns a version string."""
    driver = _mock_driver([{"version": "5.25.0"}])
    result = PreflightResult()
    await _check_apoc(driver, "neo4j", result)
    assert result.apoc_version == "5.25.0"


@pytest.mark.asyncio
async def test_apoc_check_failure() -> None:
    """Raises PreflightError when APOC is not installed."""
    driver = _mock_driver(None)
    result = PreflightResult()
    with pytest.raises(PreflightError, match="APOC Core is not available"):
        await _check_apoc(driver, "neo4j", result)


# ---------------------------------------------------------------------------
# Existing Data Warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_data_empty() -> None:
    """Reports zero nodes for empty graph."""
    driver = _mock_driver([{"count": 0}])
    result = PreflightResult()
    await _check_existing_data(driver, "neo4j", result)
    assert result.node_count == 0


@pytest.mark.asyncio
async def test_existing_data_nonempty() -> None:
    """Reports node count for non-empty graph (warning only, no error)."""
    driver = _mock_driver([{"count": 6400}])
    result = PreflightResult()
    await _check_existing_data(driver, "neo4j", result)
    assert result.node_count == 6400


# ---------------------------------------------------------------------------
# Vector Index Check
# ---------------------------------------------------------------------------


def _mock_driver_with_index(dims: int) -> AsyncMock:
    """Create a mock driver that returns a VECTOR index with given dimensions."""
    driver = AsyncMock()
    session = AsyncMock()

    index_record = {
        "type": "VECTOR",
        "name": "chunk_embeddings",
        "indexConfig": {"vector.dimensions": dims},
    }
    session.run = AsyncMock(return_value=_make_async_result([index_record]))

    driver.session = MagicMock(return_value=session)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return driver


def _mock_driver_no_indexes() -> AsyncMock:
    """Create a mock driver with no indexes."""
    driver = AsyncMock()
    session = AsyncMock()

    mock_result = MagicMock()
    mock_result.__aiter__ = lambda _self: _AsyncRecordIterator([])
    session.run = AsyncMock(return_value=mock_result)

    driver.session = MagicMock(return_value=session)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return driver


@pytest.mark.asyncio
async def test_vector_index_matching_dimensions() -> None:
    """Passes when index dimensions match expected."""
    driver = _mock_driver_with_index(1536)
    result = PreflightResult()
    await _check_vector_index(driver, "neo4j", 1536, result)
    assert result.vector_index_dimensions == 1536


@pytest.mark.asyncio
async def test_vector_index_mismatched_dimensions() -> None:
    """Raises PreflightError when index dimensions don't match."""
    driver = _mock_driver_with_index(768)
    result = PreflightResult()
    with pytest.raises(PreflightError, match="768 dimensions but embedder is configured for 1536"):
        await _check_vector_index(driver, "neo4j", 1536, result)


@pytest.mark.asyncio
async def test_vector_index_none_when_missing() -> None:
    """Returns None dimensions when no vector index exists."""
    driver = _mock_driver_no_indexes()
    result = PreflightResult()
    await _check_vector_index(driver, "neo4j", 1536, result)
    assert result.vector_index_dimensions is None


# ---------------------------------------------------------------------------
# Voyage API Check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voyage_api_valid_key() -> None:
    """Passes when Voyage API returns correct dimensions."""
    mock_embed_result = MagicMock()
    mock_embed_result.embeddings = [[0.1] * 1536]

    mock_client = MagicMock()
    mock_client.embed.return_value = mock_embed_result

    mock_voyageai = MagicMock()
    mock_voyageai.Client.return_value = mock_client

    with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
        result = PreflightResult()
        await _check_voyage_api("pa-test-key", 1536, result)
        assert result.voyage_api_valid is True

        mock_client.embed.assert_called_once_with(
            ["preflight check"],
            model="voyage-4",
            input_type="document",
            output_dimension=1536,
        )


@pytest.mark.asyncio
async def test_voyage_api_invalid_key() -> None:
    """Raises PreflightError when Voyage API call fails."""
    mock_client = MagicMock()
    mock_client.embed.side_effect = Exception("Invalid API key")

    mock_voyageai = MagicMock()
    mock_voyageai.Client.return_value = mock_client

    with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
        result = PreflightResult()
        with pytest.raises(PreflightError, match="Voyage AI API check failed"):
            await _check_voyage_api("pa-bad-key", 1536, result)


# ---------------------------------------------------------------------------
# Full Preflight Run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_preflight_checks_all_pass() -> None:
    """Full preflight passes when all individual checks succeed."""
    with (
        patch(
            "graphrag_kg_pipeline.preflight._check_neo4j_connectivity",
            new_callable=AsyncMock,
        ) as mock_neo4j,
        patch(
            "graphrag_kg_pipeline.preflight._check_apoc",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_existing_data",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_vector_index",
            new_callable=AsyncMock,
        ),
    ):
        mock_neo4j.side_effect = lambda _d, _db, r: setattr(r, "neo4j_connected", True)

        driver = AsyncMock()
        result = await run_preflight_checks(driver, "neo4j", 1536, "")
        assert isinstance(result, PreflightResult)
        assert result.using_voyage is False


@pytest.mark.asyncio
async def test_run_preflight_checks_with_voyage() -> None:
    """Full preflight includes Voyage check when API key is provided."""
    with (
        patch(
            "graphrag_kg_pipeline.preflight._check_neo4j_connectivity",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_apoc",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_existing_data",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_vector_index",
            new_callable=AsyncMock,
        ),
        patch(
            "graphrag_kg_pipeline.preflight._check_voyage_api",
            new_callable=AsyncMock,
        ) as mock_voyage,
    ):
        driver = AsyncMock()
        result = await run_preflight_checks(driver, "neo4j", 1536, "pa-test-key")
        assert result.using_voyage is True
        mock_voyage.assert_called_once_with("pa-test-key", 1536, result)


@pytest.mark.asyncio
async def test_preflight_error_is_importable() -> None:
    """Verify PreflightError is accessible from the package root."""
    from graphrag_kg_pipeline import PreflightError as PkgPreflightError

    assert PkgPreflightError is PreflightError
