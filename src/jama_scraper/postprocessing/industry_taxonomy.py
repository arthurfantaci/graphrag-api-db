"""Industry taxonomy for consolidating industry variants.

This module provides mapping from ~96 industry name variants to
19 canonical industry names, enabling proper deduplication of
Industry nodes in the knowledge graph.
"""

from typing import TYPE_CHECKING

from rapidfuzz import fuzz, process
import structlog

if TYPE_CHECKING:
    from neo4j import Driver

logger = structlog.get_logger(__name__)


# =============================================================================
# CANONICAL INDUSTRY TAXONOMY
# =============================================================================

# Maps variants to canonical names (19 industries)
INDUSTRY_TAXONOMY: dict[str, str] = {
    # Aerospace & Defense
    "aerospace": "aerospace",
    "aerospace industry": "aerospace",
    "aviation": "aerospace",
    "commercial aviation": "aerospace",
    "defense": "defense",
    "defense industry": "defense",
    "military": "defense",
    "defense & aerospace": "defense",
    "aerospace and defense": "aerospace",
    "a&d": "aerospace",
    # Automotive
    "automotive": "automotive",
    "automotive industry": "automotive",
    "automobile": "automotive",
    "auto industry": "automotive",
    "vehicle": "automotive",
    "vehicles": "automotive",
    "car": "automotive",
    "cars": "automotive",
    "autonomous vehicles": "automotive",
    "electric vehicles": "automotive",
    "ev": "automotive",
    # Medical Devices
    "medical devices": "medical devices",
    "medical device": "medical devices",
    "med device": "medical devices",
    "medtech": "medical devices",
    "medical technology": "medical devices",
    "healthcare devices": "medical devices",
    "life sciences": "life sciences",
    "pharmaceutical": "life sciences",
    "pharma": "life sciences",
    "biopharma": "life sciences",
    "biotech": "life sciences",
    "biotechnology": "life sciences",
    # Industrial
    "industrial": "industrial equipment",
    "industrial equipment": "industrial equipment",
    "industrial machinery": "industrial equipment",
    "heavy equipment": "industrial equipment",
    "manufacturing": "manufacturing",
    "manufacturing industry": "manufacturing",
    "discrete manufacturing": "manufacturing",
    "process manufacturing": "manufacturing",
    # Energy & Utilities
    "energy": "energy",
    "energy industry": "energy",
    "oil and gas": "energy",
    "oil & gas": "energy",
    "utilities": "utilities",
    "power generation": "utilities",
    "nuclear": "nuclear",
    "nuclear energy": "nuclear",
    "nuclear industry": "nuclear",
    # Transportation
    "rail": "rail",
    "railway": "rail",
    "railroad": "rail",
    "rail industry": "rail",
    "transportation": "transportation",
    "transport": "transportation",
    "logistics": "transportation",
    "marine": "marine",
    "maritime": "marine",
    "shipbuilding": "marine",
    # Technology & Electronics
    "consumer electronics": "consumer electronics",
    "electronics": "consumer electronics",
    "semiconductor": "semiconductor",
    "semiconductors": "semiconductor",
    "chip industry": "semiconductor",
    "telecommunications": "telecommunications",
    "telecom": "telecommunications",
    "telco": "telecommunications",
    "communications": "telecommunications",
    # Financial & Services
    "financial services": "financial services",
    "finance": "financial services",
    "banking": "financial services",
    "fintech": "financial services",
    # Space
    "space": "space",
    "space industry": "space",
    "satellite": "space",
    "satellites": "space",
    # Other regulated
    "government": "government",
    "public sector": "government",
    "federal": "government",
    # High-tech (distinct from traditional industries)
    "high-tech": "high-tech",
    "technology": "high-tech",
    "tech industry": "high-tech",
    "software": "software",
    "software industry": "software",
    "saas": "software",
}

# Canonical list (19 industries)
CANONICAL_INDUSTRIES = sorted(set(INDUSTRY_TAXONOMY.values()))


def normalize_industry(raw_name: str, threshold: int = 80) -> str | None:
    """Normalize an industry name to its canonical form.

    Uses exact matching first, then fuzzy matching with rapidfuzz.

    Args:
        raw_name: Raw industry name to normalize.
        threshold: Minimum fuzzy match score (0-100).

    Returns:
        Canonical industry name or None if no match.

    Example:
        >>> normalize_industry("Automotive Industry")
        'automotive'
        >>> normalize_industry("med devices")
        'medical devices'
        >>> normalize_industry("random text")
        None
    """
    if not raw_name:
        return None

    # Normalize input
    normalized = raw_name.lower().strip()

    # Exact match
    if normalized in INDUSTRY_TAXONOMY:
        return INDUSTRY_TAXONOMY[normalized]

    # Fuzzy match against taxonomy keys
    match = process.extractOne(
        normalized,
        INDUSTRY_TAXONOMY.keys(),
        scorer=fuzz.ratio,
    )

    if match and match[1] >= threshold:
        return INDUSTRY_TAXONOMY[match[0]]

    logger.warning(
        "Could not normalize industry",
        raw_name=raw_name,
        best_match=match[0] if match else None,
        score=match[1] if match else 0,
    )

    return None


class IndustryNormalizer:
    """Normalizer for consolidating Industry nodes in Neo4j.

    Provides methods to normalize industry names and consolidate
    duplicate Industry nodes in the graph.

    Example:
        >>> normalizer = IndustryNormalizer(driver)
        >>> stats = await normalizer.consolidate_industries()
        >>> print(f"Merged {stats['merged']} duplicate nodes")
    """

    def __init__(self, driver: "Driver", database: str = "neo4j") -> None:
        """Initialize the normalizer.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def get_current_industries(self) -> list[dict]:
        """Get all current Industry nodes.

        Returns:
            List of Industry node properties.
        """
        query = """
        MATCH (i:Industry)
        RETURN i.name AS name, i.display_name AS display_name,
               elementId(i) AS element_id, count{(i)-->()} AS relationship_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def consolidate_industries(self) -> dict:
        """Consolidate Industry nodes to canonical forms.

        Merges duplicate Industry nodes into canonical forms,
        preserving all relationships.

        Returns:
            Statistics about the consolidation.
        """
        stats = {
            "original_count": 0,
            "canonical_count": 0,
            "merged": 0,
            "failed": [],
        }

        # Get current industries
        industries = await self.get_current_industries()
        stats["original_count"] = len(industries)

        # Group by canonical name
        canonical_groups: dict[str, list[dict]] = {}
        unmatched = []

        for industry in industries:
            name = industry["name"]
            canonical = normalize_industry(name)

            if canonical:
                if canonical not in canonical_groups:
                    canonical_groups[canonical] = []
                canonical_groups[canonical].append(industry)
            else:
                unmatched.append(industry)

        stats["canonical_count"] = len(canonical_groups)

        # Merge duplicates
        for canonical, group in canonical_groups.items():
            if len(group) > 1:
                # Merge all into the first one
                primary = group[0]
                for duplicate in group[1:]:
                    try:
                        await self._merge_nodes(
                            primary["element_id"],
                            duplicate["element_id"],
                            canonical,
                        )
                        stats["merged"] += 1
                    except Exception as e:
                        stats["failed"].append({
                            "name": duplicate["name"],
                            "error": str(e),
                        })

        logger.info(
            "Industry consolidation complete",
            original=stats["original_count"],
            canonical=stats["canonical_count"],
            merged=stats["merged"],
        )

        return stats

    async def _merge_nodes(
        self,
        primary_id: str,
        duplicate_id: str,
        canonical_name: str,
    ) -> None:
        """Merge a duplicate node into the primary node.

        Args:
            primary_id: Element ID of the primary node.
            duplicate_id: Element ID of the duplicate node.
            canonical_name: Canonical name to set.
        """
        query = """
        MATCH (primary:Industry) WHERE elementId(primary) = $primary_id
        MATCH (dup:Industry) WHERE elementId(dup) = $duplicate_id

        // Transfer all relationships from duplicate to primary
        CALL {
            WITH primary, dup
            MATCH (dup)-[r]->()
            MERGE (primary)-[r2:TEMP_REL]->(endNode(r))
            SET r2 = properties(r)
            DELETE r
        }

        CALL {
            WITH primary, dup
            MATCH ()-[r]->(dup)
            MERGE (startNode(r))-[r2:TEMP_REL]->(primary)
            SET r2 = properties(r)
            DELETE r
        }

        // Update primary with canonical name
        SET primary.name = $canonical_name

        // Delete duplicate
        DELETE dup
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                primary_id=primary_id,
                duplicate_id=duplicate_id,
                canonical_name=canonical_name,
            )

    async def update_industry_names(self) -> int:
        """Update all Industry nodes to use canonical names.

        Returns:
            Number of nodes updated.
        """
        # Build CASE statement for all mappings
        case_statements = []
        for variant, canonical in INDUSTRY_TAXONOMY.items():
            escaped_variant = variant.replace("'", "\\'")
            escaped_canonical = canonical.replace("'", "\\'")
            case_statements.append(
                f"WHEN toLower(i.name) = '{escaped_variant}' THEN '{escaped_canonical}'"
            )

        case_expr = "\n        ".join(case_statements)

        query = f"""
        MATCH (i:Industry)
        WITH i, CASE
            {case_expr}
            ELSE i.name
        END AS canonical_name
        WHERE canonical_name <> i.name
        SET i.name = canonical_name
        RETURN count(i) AS updated
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["updated"] if record else 0
