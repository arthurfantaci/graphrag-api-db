"""Pytest configuration and shared test fixtures.

This module provides fixtures for testing the Jama Guide Scraper,
including sample HTML content, mock data, and Neo4j test utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# =============================================================================
# HTML CONTENT FIXTURES
# =============================================================================


@pytest.fixture
def sample_article_html() -> str:
    """Provide sample HTML content matching Jama's article pages.

    Returns:
        A minimal HTML structure for parser testing.
    """
    return """
    <html>
    <head><title>Test Article | Jama Software</title></head>
    <body>
        <div class="flex_cell">Navigation</div>
        <div class="flex_cell">
            <h1>Test Article Title</h1>
            <p>Test content paragraph with <strong>key concept</strong>.</p>
            <h2>Section One</h2>
            <p>Section one content about requirements traceability.</p>
            <h2>Section Two</h2>
            <p>Section two content about ISO 26262 in automotive.</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_article_html_with_headers() -> str:
    """Provide HTML with multiple header levels for chunking tests.

    Returns:
        HTML with h1, h2, h3 headers for hierarchical splitting.
    """
    return """
    <html>
    <body>
        <h1>Requirements Management Guide</h1>
        <p>Introduction paragraph about requirements management.</p>

        <h2>What is Requirements Management?</h2>
        <p>Requirements management is a systematic approach to eliciting,
        organizing, and documenting requirements. It involves tracking
        changes and ensuring alignment with project goals.</p>

        <h3>Key Benefits</h3>
        <p>The key benefits include improved communication, reduced risk,
        and better project outcomes. Teams that implement proper requirements
        management see significant improvements in delivery.</p>

        <h2>Traceability Fundamentals</h2>
        <p>Traceability is the ability to link requirements to their
        sources and to the artifacts that implement them.</p>

        <h3>Bidirectional Traceability</h3>
        <p>Bidirectional traceability allows teams to trace both forward
        from requirements to implementation and backward from implementation
        to requirements.</p>
    </body>
    </html>
    """


@pytest.fixture
def sample_glossary_html() -> str:
    """Provide sample HTML content matching Jama's glossary page.

    Returns:
        A minimal HTML structure for glossary parser testing.
    """
    return """
    <html>
    <head><title>Glossary | Jama Software</title></head>
    <body>
        <div class="flex_cell">
            <h1>Glossary</h1>
            <dl>
                <dt>Requirements Traceability</dt>
                <dd>The ability to trace requirements through the development lifecycle.</dd>
                <dt>Verification</dt>
                <dd>Ensuring the product is built correctly.</dd>
                <dt>Validation</dt>
                <dd>Ensuring the right product is built.</dd>
            </dl>
        </div>
    </body>
    </html>
    """


# =============================================================================
# MODEL FIXTURES
# =============================================================================


@pytest.fixture
def sample_article_data() -> dict:
    """Provide sample article data for model testing.

    Returns:
        Dictionary matching Article model structure.
    """
    return {
        "article_id": "ch1-art1",
        "title": "What is Requirements Management?",
        "url": "https://www.jamasoftware.com/requirements-management-guide/chapter-1/article-1",
        "content_type": "article",
        "markdown_content": "# Requirements Management\n\nRequirements management is...",
        "sections": [
            {"heading": "Introduction", "content": "Introduction content"},
            {"heading": "Key Concepts", "content": "Key concepts content"},
        ],
        "key_concepts": ["requirements management", "traceability"],
        "cross_references": [],
        "word_count": 150,
        "character_count": 850,
    }


@pytest.fixture
def sample_guide_data(sample_article_data: dict) -> dict:
    """Provide sample guide data for integration testing.

    Returns:
        Dictionary matching RequirementsManagementGuide structure.
    """
    return {
        "metadata": {
            "title": "The Essential Guide to Requirements Management",
            "publisher": "Jama Software",
            "base_url": "https://www.jamasoftware.com/requirements-management-guide/",
            "scraped_at": "2024-01-15T10:30:00Z",
            "total_chapters": 1,
        },
        "chapters": [
            {
                "chapter_number": 1,
                "title": "Requirements Management",
                "overview_url": "https://www.jamasoftware.com/requirements-management-guide/chapter-1",
                "articles": [sample_article_data],
            }
        ],
        "glossary": {
            "terms": [
                {
                    "term": "Traceability",
                    "definition": "The ability to trace requirements.",
                }
            ]
        },
    }


# =============================================================================
# EXTRACTION SCHEMA FIXTURES
# =============================================================================


@pytest.fixture
def sample_extraction_result() -> dict:
    """Provide sample LLM extraction result for testing.

    Returns:
        Dictionary matching neo4j_graphrag extraction output.
    """
    return {
        "nodes": [
            {
                "id": "0",
                "label": "Concept",
                "properties": {
                    "name": "requirements traceability",
                    "display_name": "Requirements Traceability",
                },
            },
            {
                "id": "1",
                "label": "Industry",
                "properties": {
                    "name": "automotive",
                    "display_name": "Automotive",
                    "regulated": True,
                },
            },
            {
                "id": "2",
                "label": "Standard",
                "properties": {
                    "name": "iso 26262",
                    "display_name": "ISO 26262",
                    "organization": "ISO",
                },
            },
        ],
        "relationships": [
            {
                "type": "APPLIES_TO",
                "start_node_id": "0",
                "end_node_id": "1",
                "properties": {},
            },
            {
                "type": "APPLIES_TO",
                "start_node_id": "2",
                "end_node_id": "1",
                "properties": {},
            },
        ],
    }


# =============================================================================
# INDUSTRY TAXONOMY FIXTURES
# =============================================================================


@pytest.fixture
def sample_industry_names() -> list[str]:
    """Provide sample industry names for taxonomy testing.

    Returns:
        List of industry names including variants and edge cases.
    """
    return [
        "automotive",
        "Automotive",
        "AUTOMOTIVE",
        "auto",
        "medical devices",
        "medical device",
        "medtech",
        "aerospace",
        "aerospace & defense",
        "aerospace and defense",
        "AI",
        "artificial intelligence",
        "IoT",
        "software development",
        "regulated",
        "industry",
        "TÜV SÜD",
    ]


# =============================================================================
# NEO4J MOCK FIXTURES
# =============================================================================


@dataclass
class MockRecord:
    """Mock Neo4j record for testing."""

    data: dict

    def __getitem__(self, key: str) -> any:
        """Get item from record data."""
        return self.data[key]

    def keys(self) -> list[str]:
        """Get record keys."""
        return list(self.data.keys())


class MockResult:
    """Mock Neo4j result for testing."""

    def __init__(self, records: list[dict]) -> None:
        """Initialize with list of record dicts."""
        self._records = [MockRecord(r) for r in records]
        self._index = 0

    async def single(self) -> MockRecord | None:
        """Return single record or None."""
        return self._records[0] if self._records else None

    def __aiter__(self) -> MockResult:
        """Return async iterator."""
        return self

    async def __anext__(self) -> MockRecord:
        """Get next record."""
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record


class MockSession:
    """Mock Neo4j async session for testing."""

    def __init__(self, results: dict[str, list[dict]] | None = None) -> None:
        """Initialize with query -> results mapping."""
        self._results = results or {}
        self._default_result: list[dict] = []

    def set_result(self, query_pattern: str, records: list[dict]) -> None:
        """Set result for queries matching pattern."""
        self._results[query_pattern] = records

    def set_default_result(self, records: list[dict]) -> None:
        """Set default result for unmatched queries."""
        self._default_result = records

    async def run(self, query: str, **kwargs: any) -> MockResult:
        """Run query and return mock result."""
        for pattern, records in self._results.items():
            if pattern in query:
                return MockResult(records)
        return MockResult(self._default_result)

    async def __aenter__(self) -> MockSession:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: any) -> None:
        """Exit async context."""


class MockDriver:
    """Mock Neo4j async driver for testing."""

    def __init__(self, session: MockSession | None = None) -> None:
        """Initialize with optional mock session."""
        self._session = session or MockSession()

    def session(self, database: str = "neo4j") -> MockSession:
        """Return mock session."""
        return self._session

    async def close(self) -> None:
        """Close driver (no-op for mock)."""


@pytest.fixture
def mock_neo4j_driver() -> MockDriver:
    """Provide mock Neo4j driver for testing.

    Returns:
        MockDriver instance with configurable results.
    """
    return MockDriver()


@pytest.fixture
def mock_neo4j_session() -> MockSession:
    """Provide mock Neo4j session for testing.

    Returns:
        MockSession instance with configurable results.
    """
    return MockSession()


# =============================================================================
# CHUNKING FIXTURES
# =============================================================================


@pytest.fixture
def chunking_config_dict() -> dict:
    """Provide chunking configuration for testing.

    Returns:
        Dictionary of chunking parameters.
    """
    return {
        "sliding_window_size": 512,
        "sliding_window_overlap": 64,
        "sliding_window_threshold": 1500,
        "headers_to_split_on": [
            ("h1", "article_title"),
            ("h2", "section"),
            ("h3", "subsection"),
        ],
    }


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide mock environment variables for testing.

    Returns:
        Dictionary of environment variables that were set.
    """
    env_vars = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "testpassword",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "sk-test-key-123",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    # Set optional keys to empty so load_dotenv() won't refill from .env
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    return env_vars
