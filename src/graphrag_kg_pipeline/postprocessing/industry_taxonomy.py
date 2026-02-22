"""Industry taxonomy for consolidating industry variants.

This module provides mapping from industry name variants to
canonical industry names, enabling proper deduplication of
Industry nodes in the knowledge graph.

It also identifies:
- Terms that are concepts, not industries (should be reclassified)
- Terms that are too generic (should be deleted)
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

# Maps variants to canonical names
INDUSTRY_TAXONOMY: dict[str, str] = {
    # Aerospace & Defense
    "aerospace": "aerospace",
    "aerospace industry": "aerospace",
    "aviation": "aerospace",
    "commercial aviation": "aerospace",
    "aerospace and defense": "aerospace",
    "aerospace & defense": "aerospace",
    "a&d": "aerospace",
    "defense": "defense",
    "defense industry": "defense",
    "military": "defense",
    "defense & aerospace": "defense",
    # Automotive
    "automotive": "automotive",
    "automotive industry": "automotive",
    "automobile": "automotive",
    "automobiles": "automotive",
    "auto industry": "automotive",
    "vehicle": "automotive",
    "vehicles": "automotive",
    "car": "automotive",
    "cars": "automotive",
    "autonomous vehicles": "automotive",
    "electric vehicles": "automotive",
    "ev": "automotive",
    # Medical Devices & Life Sciences
    "medical devices": "medical devices",
    "medical device": "medical devices",
    "med device": "medical devices",
    "med devices": "medical devices",
    "medtech": "medical devices",
    "medical technology": "medical devices",
    "healthcare devices": "medical devices",
    "medical": "medical devices",
    "healthcare": "healthcare",
    "health care": "healthcare",
    "life sciences": "life sciences",
    "pharmaceutical": "life sciences",
    "pharmaceuticals": "life sciences",
    "pharma": "life sciences",
    "biopharma": "life sciences",
    "biotech": "life sciences",
    "biotechnology": "life sciences",
    "pharmaceutical manufacturing": "life sciences",
    "dentistry": "healthcare",
    # Industrial & Manufacturing
    "industrial": "industrial equipment",
    "industrial equipment": "industrial equipment",
    "industrial machinery": "industrial equipment",
    "industrial manufacturing": "manufacturing",
    "heavy equipment": "industrial equipment",
    "machinery": "industrial equipment",
    "manufacturing": "manufacturing",
    "manufacturing industry": "manufacturing",
    "discrete manufacturing": "manufacturing",
    "process manufacturing": "manufacturing",
    "semiconductor manufacturing": "semiconductor",
    # Consumer Products
    "consumer electronics": "consumer electronics",
    "electronics": "consumer electronics",
    "consumer goods": "consumer goods",
    "consumer products": "consumer goods",
    "food and beverage": "consumer goods",
    "food & beverage": "consumer goods",
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
    # Technology
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
    "financial": "financial services",
    "banking": "financial services",
    "fintech": "financial services",
    "insurance": "financial services",
    # Space
    "space": "space",
    "space industry": "space",
    "space systems": "space",
    "satellite": "space",
    "satellites": "space",
    # Government & Public Sector
    "government": "government",
    "public sector": "government",
    "federal": "government",
    # Construction & Architecture
    "aec": "construction",
    "architecture": "construction",
    "construction": "construction",
    "engineering construction": "construction",
    # Software (distinct from software development concept)
    "software": "software",
    "software industry": "software",
    "saas": "software",
}

# =============================================================================
# TERMS THAT ARE CONCEPTS, NOT INDUSTRIES
# =============================================================================
# These should be reclassified from Industry to Concept nodes

CONCEPTS_NOT_INDUSTRIES: set[str] = {
    # Technology/methodology concepts
    "artificial intelligence",
    "automation",
    "digital transformation",
    "e-commerce",
    "iot",
    "internet of things",
    "machine learning",
    "ai",
    "ml",
    # Development/engineering concepts
    "software development",
    "product development",
    "systems development",
    "systems and software engineering",
    "engineering",
    "software factories",
    "workforce software",
    "consumer product development",
    # Quality/process concepts
    "quality",
    "safety",
    "safety-critical",
    "sustainability",
    "global supply chain",
    "supply chain",
}

# =============================================================================
# ORGANIZATIONS MISLABELED AS INDUSTRIES
# =============================================================================
# These are organizations that the LLM may extract as Industry nodes.
# They should be relabeled as Organization during consolidation.

ORGANIZATIONS_NOT_INDUSTRIES: set[str] = {
    # Standards bodies
    "iso",
    "iec",
    "ieee",
    "rtca",
    "sae",
    "ecss",
    "cenelec",
    "incose",
    # Regulatory agencies
    "fda",
    "faa",
    "easa",
    # Certification bodies
    "tüv süd",
    "tuv sud",
    "tüv rheinland",
    "ul",
    "sgs",
    "bureau veritas",
    "intertek",
    # Professional societies / other organizations
    "nasa",
    "pmi",
    "jama software",
    "nikola",
    "finnish red cross",
}

# =============================================================================
# GENERIC TERMS TO DELETE
# =============================================================================
# These are too vague to be useful as Industry nodes

GENERIC_TERMS_TO_DELETE: set[str] = {
    "industry",
    "industries",
    "general",
    "regulated",
    "regulated industry",
    "regulated industries",
    "regulated products",
    "multiple industries",
    "various industries",
    "other industries",
    "smbs",
    "ffrdc",
}

# Canonical list of industries
CANONICAL_INDUSTRIES = sorted(set(INDUSTRY_TAXONOMY.values()))


def classify_industry_term(raw_name: str) -> tuple[str, str | None]:
    """Classify an industry term and return its disposition.

    Args:
        raw_name: Raw industry name to classify.

    Returns:
        Tuple of (action, value) where action is one of:
        - ("keep", canonical_name): Valid industry, normalize to canonical
        - ("reclassify", None): Should be reclassified as Concept
        - ("reclassify_org", None): Should be reclassified as Organization
        - ("delete", None): Too generic, should be deleted
        - ("unknown", None): Could not classify
    """
    if not raw_name:
        return ("delete", None)

    normalized = raw_name.lower().strip()

    # Check if it's an organization that was misclassified
    if normalized in ORGANIZATIONS_NOT_INDUSTRIES:
        return ("reclassify_org", None)

    # Check if it's a concept that was misclassified
    if normalized in CONCEPTS_NOT_INDUSTRIES:
        return ("reclassify", None)

    # Check if it's a generic term to delete
    if normalized in GENERIC_TERMS_TO_DELETE:
        return ("delete", None)

    # Check for exact match in taxonomy
    if normalized in INDUSTRY_TAXONOMY:
        return ("keep", INDUSTRY_TAXONOMY[normalized])

    # Fuzzy match against taxonomy keys
    match = process.extractOne(
        normalized,
        INDUSTRY_TAXONOMY.keys(),
        scorer=fuzz.ratio,
    )

    if match and match[1] >= 80:
        return ("keep", INDUSTRY_TAXONOMY[match[0]])

    # Check fuzzy match against concepts (lower threshold)
    concept_match = process.extractOne(
        normalized,
        CONCEPTS_NOT_INDUSTRIES,
        scorer=fuzz.ratio,
    )

    if concept_match and concept_match[1] >= 75:
        return ("reclassify", None)

    # Check fuzzy match against generic terms
    generic_match = process.extractOne(
        normalized,
        GENERIC_TERMS_TO_DELETE,
        scorer=fuzz.ratio,
    )

    if generic_match and generic_match[1] >= 75:
        return ("delete", None)

    return ("unknown", None)


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

    # Check if it should be reclassified or deleted
    if normalized in CONCEPTS_NOT_INDUSTRIES:
        return None
    if normalized in GENERIC_TERMS_TO_DELETE:
        return None

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

    Provides methods to:
    - Reclassify misidentified Industry nodes to Concept
    - Delete overly generic Industry nodes
    - Consolidate remaining industries to canonical forms

    Example:
        >>> normalizer = IndustryNormalizer(driver)
        >>> stats = await normalizer.consolidate_industries()
        >>> print(f"Reclassified {stats['reclassified']}, deleted {stats['deleted']}")
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
               elementId(i) AS element_id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def consolidate_industries(self) -> dict:
        """Consolidate Industry nodes: reclassify, delete, and merge.

        This method:
        1. Reclassifies concept terms from Industry to Concept nodes
        2. Deletes generic/meaningless Industry nodes
        3. Merges remaining industries to canonical forms

        Returns:
            Statistics about the consolidation.
        """
        stats = {
            "original_count": 0,
            "reclassified": 0,
            "deleted": 0,
            "merged": 0,
            "canonical_count": 0,
            "unknown": [],
        }

        # Get current industries
        industries = await self.get_current_industries()
        stats["original_count"] = len(industries)

        # Classify each industry
        to_reclassify = []
        to_reclassify_org = []
        to_delete = []
        canonical_groups: dict[str, list[dict]] = {}
        unknown = []

        for industry in industries:
            name = industry["name"]
            action, canonical = classify_industry_term(name)

            if action == "reclassify":
                to_reclassify.append(industry)
            elif action == "reclassify_org":
                to_reclassify_org.append(industry)
            elif action == "delete":
                to_delete.append(industry)
            elif action == "keep" and canonical:
                if canonical not in canonical_groups:
                    canonical_groups[canonical] = []
                canonical_groups[canonical].append(industry)
            else:
                unknown.append(industry)

        # Step 1a: Reclassify concepts
        for industry in to_reclassify:
            try:
                await self._reclassify_to_concept(industry["element_id"])
                stats["reclassified"] += 1
                logger.debug("Reclassified to Concept", name=industry["name"])
            except Exception as e:
                logger.error(
                    "Failed to reclassify",
                    name=industry["name"],
                    error=str(e),
                )

        # Step 1b: Reclassify organizations
        for industry in to_reclassify_org:
            try:
                await self._reclassify_to_organization(industry["element_id"])
                stats["reclassified"] += 1
                logger.debug("Reclassified to Organization", name=industry["name"])
            except Exception as e:
                logger.error(
                    "Failed to reclassify to Organization",
                    name=industry["name"],
                    error=str(e),
                )

        # Step 2: Delete generic terms
        for industry in to_delete:
            try:
                await self._delete_industry(industry["element_id"])
                stats["deleted"] += 1
                logger.debug("Deleted generic term", name=industry["name"])
            except Exception as e:
                logger.error(
                    "Failed to delete",
                    name=industry["name"],
                    error=str(e),
                )

        # Step 3: Merge duplicates within canonical groups
        # Order: merge all duplicates first, then update name
        for canonical, group in canonical_groups.items():
            if len(group) > 1:
                primary = group[0]
                # First merge all duplicates into primary
                for duplicate in group[1:]:
                    try:
                        await self._merge_industry_nodes(
                            primary["element_id"],
                            duplicate["element_id"],
                        )
                        stats["merged"] += 1
                    except Exception as e:
                        logger.error(
                            "Failed to merge",
                            primary=primary["name"],
                            duplicate=duplicate["name"],
                            error=str(e),
                        )
                # After merging, update primary to canonical name
                try:
                    await self._update_industry_name(
                        primary["element_id"],
                        canonical,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to update name",
                        original=primary["name"],
                        canonical=canonical,
                        error=str(e),
                    )
            elif len(group) == 1:
                # Single node, just update name to canonical
                try:
                    await self._update_industry_name(
                        group[0]["element_id"],
                        canonical,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to update name",
                        original=group[0]["name"],
                        canonical=canonical,
                        error=str(e),
                    )

        stats["canonical_count"] = len(canonical_groups)
        stats["unknown"] = [u["name"] for u in unknown]

        logger.info(
            "Industry consolidation complete",
            original=stats["original_count"],
            reclassified=stats["reclassified"],
            deleted=stats["deleted"],
            merged=stats["merged"],
            canonical=stats["canonical_count"],
            unknown=len(unknown),
        )

        return stats

    async def _reclassify_to_concept(self, element_id: str) -> None:
        """Reclassify an Industry node to a Concept node.

        Args:
            element_id: Element ID of the Industry node.
        """
        query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        // Check if a Concept with the same name exists
        OPTIONAL MATCH (existing:Concept {name: i.name})
        WITH i, existing
        WHERE existing IS NULL
        // Add Concept label, remove Industry label
        SET i:Concept
        REMOVE i:Industry
        """

        merge_query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        MATCH (existing:Concept {name: i.name})
        // Transfer relationships to existing Concept
        CALL (i, existing) {
            MATCH (i)-[r]->()
            WITH i, existing, r, endNode(r) AS target, type(r) AS rel_type
            CALL apoc.merge.relationship(existing, rel_type, {}, properties(r), target, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS out_count
        }
        CALL (i, existing) {
            MATCH ()-[r]->(i)
            WITH i, existing, r, startNode(r) AS source, type(r) AS rel_type
            CALL apoc.merge.relationship(source, rel_type, {}, properties(r), existing, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS in_count
        }
        DELETE i
        """

        async with self.driver.session(database=self.database) as session:
            # First try to just relabel
            result = await session.run(query, element_id=element_id)
            summary = await result.consume()

            # If no nodes were modified, merge with existing Concept
            if summary.counters.labels_added == 0:
                await session.run(merge_query, element_id=element_id)

    async def _reclassify_to_organization(self, element_id: str) -> None:
        """Reclassify an Industry node to an Organization node.

        Args:
            element_id: Element ID of the Industry node.
        """
        query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        // Check if an Organization with the same name exists
        OPTIONAL MATCH (existing:Organization {name: i.name})
        WITH i, existing
        WHERE existing IS NULL
        // Add Organization label, remove Industry label
        SET i:Organization
        REMOVE i:Industry
        """

        merge_query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        MATCH (existing:Organization {name: i.name})
        // Transfer relationships to existing Organization
        CALL (i, existing) {
            MATCH (i)-[r]->()
            WITH i, existing, r, endNode(r) AS target, type(r) AS rel_type
            CALL apoc.merge.relationship(existing, rel_type, {}, properties(r), target, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS out_count
        }
        CALL (i, existing) {
            MATCH ()-[r]->(i)
            WITH i, existing, r, startNode(r) AS source, type(r) AS rel_type
            CALL apoc.merge.relationship(source, rel_type, {}, properties(r), existing, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS in_count
        }
        DELETE i
        """

        async with self.driver.session(database=self.database) as session:
            # First try to just relabel
            result = await session.run(query, element_id=element_id)
            summary = await result.consume()

            # If no nodes were modified, merge with existing Organization
            if summary.counters.labels_added == 0:
                await session.run(merge_query, element_id=element_id)

    async def _delete_industry(self, element_id: str) -> None:
        """Delete an Industry node and its relationships.

        Args:
            element_id: Element ID of the Industry node.
        """
        query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        DETACH DELETE i
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query, element_id=element_id)

    async def _update_industry_name(
        self,
        element_id: str,
        canonical_name: str,
    ) -> None:
        """Update an Industry node's name to canonical form.

        Args:
            element_id: Element ID of the Industry node.
            canonical_name: Canonical name to set.
        """
        query = """
        MATCH (i:Industry) WHERE elementId(i) = $element_id
        SET i.name = $canonical_name
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                element_id=element_id,
                canonical_name=canonical_name,
            )

    async def _merge_industry_nodes(
        self,
        primary_id: str,
        duplicate_id: str,
    ) -> None:
        """Merge a duplicate Industry node into the primary node.

        Args:
            primary_id: Element ID of the primary node.
            duplicate_id: Element ID of the duplicate node.
        """
        query = """
        MATCH (primary:Industry) WHERE elementId(primary) = $primary_id
        MATCH (dup:Industry) WHERE elementId(dup) = $duplicate_id

        // Transfer outgoing relationships
        CALL (primary, dup) {
            MATCH (dup)-[r]->()
            WITH primary, r, endNode(r) AS target, type(r) AS rel_type
            CALL apoc.merge.relationship(primary, rel_type, {}, properties(r), target, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS out_count
        }

        // Transfer incoming relationships
        CALL (primary, dup) {
            MATCH ()-[r]->(dup)
            WITH primary, r, startNode(r) AS source, type(r) AS rel_type
            CALL apoc.merge.relationship(source, rel_type, {}, properties(r), primary, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS in_count
        }

        // Merge non-name properties (keep primary's name, avoid constraint violation)
        WITH primary, dup, [key IN keys(dup) WHERE key <> 'name' | [key, dup[key]]] AS props
        UNWIND props AS prop
        SET primary[prop[0]] = CASE
            WHEN primary[prop[0]] IS NULL THEN prop[1]
            ELSE primary[prop[0]]
        END

        // Delete duplicate
        WITH primary, dup
        DELETE dup
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                primary_id=primary_id,
                duplicate_id=duplicate_id,
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
