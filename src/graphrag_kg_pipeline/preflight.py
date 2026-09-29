"""Pre-flight validation for the GraphRAG ingestion pipeline.

Runs automated checks before starting a potentially long ingestion run
to catch configuration and connectivity issues early. All checks are
fail-fast: the first critical failure aborts with a clear error message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from typesafe_sdk import Noul

from graphrag_kg_pipeline.utils.typesafe_client import create_typesafe_client

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)


class PreflightError(Exception):
    """Pre-flight validation failed.

    Raised when a critical check fails during pre-flight validation.
    Contains a user-friendly message describing the issue and how to fix it.
    """


@dataclass
class PreflightResult:
    """Result of all pre-flight checks.

    Attributes:
        neo4j_connected: Whether Neo4j is reachable.
        apoc_version: APOC Core version string, or empty if unavailable.
        node_count: Number of existing nodes in the database.
        vector_index_dimensions: Dimensions of existing vector index, or None if no index.
        voyage_api_valid: Whether the Voyage API key produced a valid embedding.
        using_voyage: Whether Voyage AI will be used for embeddings.
        typesafe_api_valid: Whether the TypeSafe API key answered a test question.
    """

    neo4j_connected: bool = False
    apoc_version: str = ""
    node_count: int = 0
    vector_index_dimensions: int | None = None
    voyage_api_valid: bool = False
    using_voyage: bool = False
    typesafe_api_valid: bool = False


async def run_preflight_checks(
    driver: AsyncDriver,
    database: str = "neo4j",
    expected_dimensions: int = 1536,
    voyage_api_key: str = "",
    typesafe_api_key: str = "",
) -> PreflightResult:
    """Run all pre-flight validation checks.

    Checks are run sequentially — the first critical failure raises PreflightError.
    Non-critical issues (e.g., existing data) are logged as warnings.

    Args:
        driver: Async Neo4j driver.
        database: Neo4j database name.
        expected_dimensions: Expected embedding vector dimensions.
        voyage_api_key: Voyage AI API key (empty string to skip check).
        typesafe_api_key: TypeSafe API key. Required; industry consolidation
            cannot run without it, so an empty value fails before any network call.

    Returns:
        PreflightResult with check outcomes.

    Raises:
        PreflightError: If a critical check fails.
    """
    result = PreflightResult(using_voyage=bool(voyage_api_key))

    # 0. TypeSafe key presence (cheapest check; consolidation needs it)
    if not typesafe_api_key.strip():
        msg = (
            "TYPESAFE_API_KEY is not set. Industry consolidation needs a TypeSafe "
            "System One judgment for every name the exact-match tables miss.\n"
            "Add TYPESAFE_API_KEY to your .env file or export it in the shell."
        )
        raise PreflightError(msg)

    # 1. Neo4j connectivity
    await _check_neo4j_connectivity(driver, database, result)

    # 2. APOC availability
    await _check_apoc(driver, database, result)

    # 3. Existing data warning
    await _check_existing_data(driver, database, result)

    # 4. Vector index dimensions
    await _check_vector_index(driver, database, expected_dimensions, result)

    # 5. Voyage API key validity
    if voyage_api_key:
        await _check_voyage_api(voyage_api_key, expected_dimensions, result)

    # 6. TypeSafe API key validity
    await _check_typesafe_api(typesafe_api_key, result)

    logger.info(
        "Pre-flight checks passed",
        neo4j=result.neo4j_connected,
        apoc=result.apoc_version,
        node_count=result.node_count,
        vector_index_dims=result.vector_index_dimensions,
        using_voyage=result.using_voyage,
        voyage_valid=result.voyage_api_valid,
        typesafe_valid=result.typesafe_api_valid,
    )

    return result


async def _check_neo4j_connectivity(
    driver: AsyncDriver,
    database: str,
    result: PreflightResult,
) -> None:
    """Verify the Neo4j database is reachable.

    Args:
        driver: Async Neo4j driver.
        database: Database name.
        result: PreflightResult to update.

    Raises:
        PreflightError: If the database is unreachable.
    """
    try:
        async with driver.session(database=database) as session:
            records = await session.run("RETURN 1 AS ping")
            record = await records.single()
            if record and record["ping"] == 1:
                result.neo4j_connected = True
                logger.info("Neo4j connectivity check passed")
                return
    except Exception as e:
        msg = (
            f"Cannot connect to Neo4j: {e}\n"
            "Check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file."
        )
        raise PreflightError(msg) from e

    msg = "Neo4j returned unexpected response to ping query."
    raise PreflightError(msg)


async def _check_apoc(
    driver: AsyncDriver,
    database: str,
    result: PreflightResult,
) -> None:
    """Verify APOC Core is installed.

    APOC is required by EntityCleanupNormalizer (apoc.merge.relationship)
    and neo4j_graphrag (apoc.create.addLabels).

    Args:
        driver: Async Neo4j driver.
        database: Database name.
        result: PreflightResult to update.

    Raises:
        PreflightError: If APOC is not available.
    """
    try:
        async with driver.session(database=database) as session:
            records = await session.run("RETURN apoc.version() AS version")
            record = await records.single()
            if record:
                result.apoc_version = record["version"]
                logger.info("APOC check passed", version=result.apoc_version)
                return
    except Exception as e:
        msg = (
            f"APOC Core is not available: {e}\n"
            "APOC is required by the pipeline (apoc.create.addLabels, "
            "apoc.merge.relationship). On Aura Professional, APOC Core "
            "is pre-installed. For self-hosted, install the APOC plugin."
        )
        raise PreflightError(msg) from e

    msg = "APOC version query returned no result."
    raise PreflightError(msg)


async def _check_existing_data(
    driver: AsyncDriver,
    database: str,
    result: PreflightResult,
) -> None:
    """Check for existing data and warn if the graph is not empty.

    This is a non-critical warning — the pipeline can still run, but
    re-ingestion without clearing first may create duplicates.

    Args:
        driver: Async Neo4j driver.
        database: Database name.
        result: PreflightResult to update.
    """
    async with driver.session(database=database) as session:
        records = await session.run("MATCH (n) RETURN count(n) AS count")
        record = await records.single()
        if record:
            result.node_count = record["count"]

    if result.node_count > 0:
        logger.warning(
            "Database contains existing data — re-ingestion may create duplicates. "
            "Clear with: MATCH (n) DETACH DELETE n",
            node_count=result.node_count,
        )


async def _check_vector_index(
    driver: AsyncDriver,
    database: str,
    expected_dimensions: int,
    result: PreflightResult,
) -> None:
    """Check existing vector index dimensions for consistency.

    If a vector index exists with different dimensions than the configured
    embedder, the pipeline will fail silently or produce unusable embeddings.

    Args:
        driver: Async Neo4j driver.
        database: Database name.
        expected_dimensions: Expected embedding dimensions from config.
        result: PreflightResult to update.

    Raises:
        PreflightError: If an existing index has mismatched dimensions.
    """
    async with driver.session(database=database) as session:
        records = await session.run("SHOW INDEXES")
        async for record in records:
            record_dict = dict(record)
            if record_dict.get("type") == "VECTOR":
                index_config = record_dict.get("indexConfig", {})
                dims = index_config.get("vector.dimensions")
                if dims is not None:
                    result.vector_index_dimensions = int(dims)
                    if result.vector_index_dimensions != expected_dimensions:
                        index_name = record_dict.get("name", "unknown")
                        msg = (
                            f"Vector index '{index_name}' has {result.vector_index_dimensions} "
                            f"dimensions but embedder is configured for {expected_dimensions}. "
                            f"Drop the index with: DROP INDEX {index_name}"
                        )
                        raise PreflightError(msg)
                    logger.info(
                        "Vector index dimensions match",
                        dimensions=result.vector_index_dimensions,
                    )
                    return

    logger.info("No existing vector index found — one will be created during ingestion")


async def _check_voyage_api(
    voyage_api_key: str,
    expected_dimensions: int,
    result: PreflightResult,
) -> None:
    """Validate the Voyage API key with a test embedding call.

    Args:
        voyage_api_key: Voyage AI API key.
        expected_dimensions: Expected output dimensions.
        result: PreflightResult to update.

    Raises:
        PreflightError: If the API key is invalid or the call fails.
    """
    try:
        import voyageai  # noqa: PLC0415

        client = voyageai.Client(api_key=voyage_api_key)
        test_result = client.embed(
            ["preflight check"],
            model="voyage-4",
            input_type="document",
            output_dimension=expected_dimensions,
        )
    except Exception as e:
        msg = (
            f"Voyage AI API check failed: {e}\n"
            "Verify VOYAGE_API_KEY in your .env file, or remove it to use OpenAI embeddings."
        )
        raise PreflightError(msg) from e

    if test_result.embeddings and len(test_result.embeddings[0]) == expected_dimensions:
        result.voyage_api_valid = True
        logger.info(
            "Voyage AI API check passed",
            dimensions=len(test_result.embeddings[0]),
        )
        return

    actual_dims = len(test_result.embeddings[0]) if test_result.embeddings else 0
    msg = f"Voyage AI returned {actual_dims} dimensions but expected {expected_dimensions}."
    raise PreflightError(msg)


async def _check_typesafe_api(api_key: str, result: PreflightResult) -> None:
    """Validate the TypeSafe API key with one tiny System One call.

    Args:
        api_key: TypeSafe API key.
        result: PreflightResult to update.

    Raises:
        PreflightError: If the client cannot be built or the call fails.
    """
    try:
        client = create_typesafe_client(api_key)
        async with client:
            response = await client.system_one(
                state="preflight check",
                questions={"english": Noul(instructions="Is the text written in English?")},
            )
    except Exception as e:
        msg = (
            f"TypeSafe API check failed: {e}\n"
            "Verify TYPESAFE_API_KEY in your .env file or shell environment."
        )
        raise PreflightError(msg) from e

    result.typesafe_api_valid = True
    logger.info("TypeSafe API check passed", model=response.model)
