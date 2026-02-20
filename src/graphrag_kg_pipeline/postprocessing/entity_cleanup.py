"""Entity cleanup taxonomy for deduplication and quality improvement.

This module provides mapping for normalizing entity names and identifying
entities that should be removed from the knowledge graph.

It handles:
- Plural → singular normalization for LLM-extracted entities
- Identification of overly generic terms that provide no semantic value
- Classification of entities for cleanup operations

IMPORTANT: This cleanup only affects LLM-extracted entity nodes:
    Concept, Tool, Role, Artifact, Processstage, Bestpractice,
    Methodology, Standard, Challenge, Industry

It does NOT affect:
    Definition (glossary), Article, Chapter, Chunk, Image, Video, Webinar

Definition nodes intentionally allow variants like "Requirement" and
"Requirements" as separate glossary terms with distinct definitions.
"""

from typing import TYPE_CHECKING

import structlog

from graphrag_kg_pipeline.extraction.schema import LLM_EXTRACTED_ENTITY_LABELS

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Driver

    # Accept either sync or async driver
    AnyDriver = Driver | AsyncDriver

logger = structlog.get_logger(__name__)


# =============================================================================
# GENERIC TERMS TO DELETE
# =============================================================================
# Terms that are too vague to be useful as entity nodes.
# These provide no semantic value and clutter the graph.

GENERIC_TERMS_TO_DELETE: set[str] = {
    # Generic tool/software terms
    "tool",
    "tools",
    "software",
    "solution",
    "solutions",
    "platform",
    "platforms",
    "system",
    "systems",
    "application",
    "applications",
    "product",
    "products",
    # Generic process/method terms
    "method",
    "methods",
    "process",
    "processes",
    "approach",
    "approaches",
    "technique",
    "techniques",
    "practice",
    "practices",
    "procedure",
    "procedures",
    # Generic document terms
    "document",
    "documents",
    "file",
    "files",
    "report",
    "reports",
    # Generic role terms
    "person",
    "people",
    "user",
    "users",
    "team",
    "teams",
    "member",
    "members",
    # Other overly generic terms
    "thing",
    "things",
    "item",
    "items",
    "element",
    "elements",
    "component",
    "components",
    "part",
    "parts",
    "type",
    "types",
    "kind",
    "kinds",
    "way",
    "ways",
    "step",
    "steps",
    "stage",
    "stages",
    "phase",
    "phases",
    "level",
    "levels",
    "area",
    "areas",
    "aspect",
    "aspects",
    "factor",
    "factors",
    "feature",
    "features",
    "function",
    "functions",
    "activity",
    "activities",
    "task",
    "tasks",
    "action",
    "actions",
    "work",
    "result",
    "results",
    "outcome",
    "outcomes",
    "output",
    "outputs",
    "input",
    "inputs",
    "data",
    "information",
    "content",
    "resource",
    "resources",
    "material",
    "materials",
}


# =============================================================================
# PLURAL TO SINGULAR MAPPING
# =============================================================================
# Maps common plural forms to their singular canonical form.
# Used for normalizing entity names during extraction and cleanup.

PLURAL_TO_SINGULAR: dict[str, str] = {
    # Requirements domain
    "requirements": "requirement",
    "specifications": "specification",
    "constraints": "constraint",
    "baselines": "baseline",
    "traceabilities": "traceability",
    "dependencies": "dependency",
    "attributes": "attribute",
    "properties": "property",
    "criteria": "criterion",
    # Stakeholders and roles
    "stakeholders": "stakeholder",
    "engineers": "engineer",
    "developers": "developer",
    "analysts": "analyst",
    "architects": "architect",
    "testers": "tester",
    "reviewers": "reviewer",
    "managers": "manager",
    "customers": "customer",
    "suppliers": "supplier",
    # Artifacts
    "artifacts": "artifact",
    "deliverables": "deliverable",
    "diagrams": "diagram",
    "models": "model",
    "prototypes": "prototype",
    "templates": "template",
    "checklists": "checklist",
    "matrices": "matrix",
    # Standards and methodologies
    "standards": "standard",
    "regulations": "regulation",
    "guidelines": "guideline",
    "frameworks": "framework",
    "methodologies": "methodology",
    "workflows": "workflow",
    # Challenges and risks
    "challenges": "challenge",
    "risks": "risk",
    "issues": "issue",
    "defects": "defect",
    "bugs": "bug",
    "errors": "error",
    "failures": "failure",
    # Tests and verification
    "tests": "test",
    "cases": "case",
    "scenarios": "scenario",
    "reviews": "review",
    "inspections": "inspection",
    "audits": "audit",
    "validations": "validation",
    "verifications": "verification",
    # Tools (when specific)
    "integrations": "integration",
    "interfaces": "interface",
    "apis": "api",
    "plugins": "plugin",
    "extensions": "extension",
    "modules": "module",
    # Other domain-specific
    "changes": "change",
    "updates": "update",
    "versions": "version",
    "releases": "release",
    "iterations": "iteration",
    "sprints": "sprint",
    "milestones": "milestone",
    "objectives": "objective",
    "goals": "goal",
    "metrics": "metric",
    "measurements": "measurement",
    "assessments": "assessment",
    "evaluations": "evaluation",
    "decisions": "decision",
    "approvals": "approval",
    "notifications": "notification",
    "alerts": "alert",
    "warnings": "warning",
}


# Words indicating positive outcomes/goals — NOT challenges
POSITIVE_OUTCOME_WORDS: frozenset[str] = frozenset(
    {
        "high-quality",
        "quality",
        "satisfaction",
        "success",
        "successful",
        "efficient",
        "efficiency",
        "effective",
        "effectiveness",
        "improved",
        "improvement",
        "reduced",
        "reduction",
        "faster",
        "better",
        "optimal",
        "reliable",
        "reliability",
        "safe",
        "safety",
        "secure",
        "security",
        "compliant",
        "compliance",
        "innovation",
        "innovative",
        "productivity",
        "performance",
        "achievement",
        "benefit",
        "advantage",
    }
)


def is_potentially_mislabeled_challenge(name: str) -> bool:
    """Check if a Challenge entity name suggests a positive outcome, not a challenge.

    Uses first-word matching against POSITIVE_OUTCOME_WORDS to detect
    mislabeled entities like "High-Quality Products" classified as Challenge.

    Args:
        name: Entity name to check.

    Returns:
        True if the entity is likely mislabeled as a Challenge.
    """
    if not name:
        return False
    words = name.lower().strip().split()
    if not words:
        return False
    # Check first word (whole-word match per Agent #3 recommendation)
    return words[0] in POSITIVE_OUTCOME_WORDS


def is_generic_term(name: str) -> bool:
    """Check if a term is too generic to be useful.

    Args:
        name: Entity name to check.

    Returns:
        True if the term is generic and should be deleted.
    """
    if not name:
        return True
    return name.lower().strip() in GENERIC_TERMS_TO_DELETE


def normalize_to_singular(name: str) -> str:
    """Normalize a plural entity name to singular form.

    Args:
        name: Entity name to normalize.

    Returns:
        Singular form if a mapping exists, otherwise original name.

    Example:
        >>> normalize_to_singular("requirements")
        'requirement'
        >>> normalize_to_singular("traceability")
        'traceability'
    """
    if not name:
        return name
    normalized = name.lower().strip()
    return PLURAL_TO_SINGULAR.get(normalized, normalized)


def classify_entity_for_cleanup(
    name: str,
    label: str,
) -> tuple[str, str | None]:
    """Classify an entity and return its cleanup disposition.

    Args:
        name: Entity name.
        label: Entity label (e.g., 'Concept', 'Tool').

    Returns:
        Tuple of (action, value) where action is one of:
        - ("keep", canonical_name): Valid entity, optionally normalized
        - ("delete", None): Too generic, should be deleted
        - ("skip", None): Not an LLM-extracted entity, skip cleanup
    """
    if not name:
        return ("delete", None)

    # Skip non-LLM-extracted entities
    if label not in LLM_EXTRACTED_ENTITY_LABELS:
        return ("skip", None)

    normalized = name.lower().strip()

    # Check if it's a generic term to delete
    if normalized in GENERIC_TERMS_TO_DELETE:
        return ("delete", None)

    # Check if it needs singular normalization
    if normalized in PLURAL_TO_SINGULAR:
        return ("keep", PLURAL_TO_SINGULAR[normalized])

    return ("keep", normalized)


class EntityCleanupClassifier:
    """Classifier for entity cleanup operations.

    Provides batch classification of entities for cleanup operations
    with statistics tracking.

    Example:
        >>> classifier = EntityCleanupClassifier()
        >>> entities = [{"name": "tools", "label": "Tool"}, ...]
        >>> result = classifier.classify_batch(entities)
        >>> print(f"To delete: {len(result['to_delete'])}")
    """

    def __init__(
        self,
        generic_terms: set[str] | None = None,
        plural_mapping: dict[str, str] | None = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            generic_terms: Custom set of generic terms. Defaults to module constant.
            plural_mapping: Custom plural→singular mapping. Defaults to module constant.
        """
        self.generic_terms = generic_terms or GENERIC_TERMS_TO_DELETE
        self.plural_mapping = plural_mapping or PLURAL_TO_SINGULAR

    def classify_batch(
        self,
        entities: list[dict],
    ) -> dict[str, list[dict]]:
        """Classify a batch of entities for cleanup.

        Args:
            entities: List of entity dicts with 'name' and 'label' keys.

        Returns:
            Dictionary with keys:
            - 'to_delete': Entities that should be removed
            - 'to_normalize': Entities that need singular normalization
            - 'to_keep': Entities that are already clean
            - 'skipped': Non-LLM-extracted entities
        """
        result: dict[str, list[dict]] = {
            "to_delete": [],
            "to_normalize": [],
            "to_keep": [],
            "skipped": [],
        }

        for entity in entities:
            name = entity.get("name", "")
            label = entity.get("label", "")

            action, normalized = classify_entity_for_cleanup(name, label)

            if action == "delete":
                result["to_delete"].append(entity)
            elif action == "skip":
                result["skipped"].append(entity)
            elif action == "keep":
                if normalized and normalized != name.lower().strip():
                    entity["normalized_name"] = normalized
                    result["to_normalize"].append(entity)
                else:
                    result["to_keep"].append(entity)

        logger.info(
            "Entity classification complete",
            to_delete=len(result["to_delete"]),
            to_normalize=len(result["to_normalize"]),
            to_keep=len(result["to_keep"]),
            skipped=len(result["skipped"]),
        )

        return result


class EntityCleanupNormalizer:
    """Normalizer for cleaning up entities in Neo4j.

    Provides methods to:
    - Delete overly generic entity nodes
    - Merge plural variants into singular forms
    - Generate cleanup preview reports

    Example:
        >>> normalizer = EntityCleanupNormalizer(driver)
        >>> preview = await normalizer.preview_cleanup()
        >>> print(f"Would delete {preview['to_delete_count']} entities")
        >>> stats = await normalizer.run_cleanup()
    """

    def __init__(self, driver: "AnyDriver", database: str = "neo4j") -> None:
        """Initialize the normalizer.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database
        self.classifier = EntityCleanupClassifier()

    async def get_cleanup_candidates(self) -> list[dict]:
        """Get all entities that are candidates for cleanup.

        Returns:
            List of entity dicts with name, label, and relationship counts.
        """
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in LLM_EXTRACTED_ENTITY_LABELS) + "]"

        query = f"""
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN {label_list})
        OPTIONAL MATCH (n)-[r]-()
        WITH n, labels(n)[0] AS node_label, count(DISTINCT r) AS relationship_count
        RETURN n.name AS name, node_label AS label, relationship_count, elementId(n) AS element_id
        ORDER BY node_label, name
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def preview_cleanup(self) -> dict:
        """Generate a preview of what cleanup would do.

        Returns:
            Dictionary with cleanup preview statistics and details.
        """
        # Get generic entities to delete using classifier
        entities = await self.get_cleanup_candidates()
        classified = self.classifier.classify_batch(entities)

        # Calculate delete impact
        delete_impact = sum(e.get("relationship_count", 0) for e in classified["to_delete"])

        # Get plural/singular pairs dynamically (includes multi-word phrases)
        plural_pairs = await self.find_plural_singular_pairs()

        # Format pairs for preview display
        to_normalize = [
            {
                "label": pair["label"],
                "name": pair["plural_name"],
                "normalized_name": pair["singular_name"],
                "relationship_count": 0,  # Would need extra query to get this
            }
            for pair in plural_pairs
        ]

        return {
            "to_delete_count": len(classified["to_delete"]),
            "to_delete": classified["to_delete"],
            "delete_relationship_impact": delete_impact,
            "to_normalize_count": len(plural_pairs),
            "to_normalize": to_normalize,
            "normalize_relationship_impact": 0,  # Relationships are transferred, not lost
            "to_keep_count": len(classified["to_keep"]),
            "skipped_count": len(classified["skipped"]),
        }

    async def delete_generic_entities(self) -> int:
        """Delete all generic entities from the graph.

        Returns:
            Number of entities deleted.
        """
        term_list = "[" + ", ".join(f"'{term}'" for term in GENERIC_TERMS_TO_DELETE) + "]"
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in LLM_EXTRACTED_ENTITY_LABELS) + "]"

        query = f"""
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN {label_list})
        AND toLower(n.name) IN {term_list}
        WITH n, n.name AS name
        DETACH DELETE n
        RETURN count(*) AS deleted_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            deleted = record["deleted_count"] if record else 0

        logger.info("Deleted generic entities", count=deleted)
        return deleted

    async def find_plural_singular_pairs(self) -> list[dict]:
        """Find all plural/singular entity pairs dynamically.

        Discovers pairs where plural.name = singular.name + 's', including
        multi-word phrases like "functional requirement" / "functional requirements".

        Returns:
            List of pairs with singular_name, plural_name, label, and element IDs.
        """
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in LLM_EXTRACTED_ENTITY_LABELS) + "]"

        query = f"""
        MATCH (singular)
        WHERE any(label IN labels(singular) WHERE label IN {label_list})
        AND singular.name IS NOT NULL
        AND NOT singular.name ENDS WITH 's'

        MATCH (plural)
        WHERE any(label IN labels(plural) WHERE label IN {label_list})
        AND plural.name = singular.name + 's'
        AND labels(singular)[0] = labels(plural)[0]

        RETURN labels(singular)[0] AS label,
               singular.name AS singular_name,
               plural.name AS plural_name,
               elementId(singular) AS singular_id,
               elementId(plural) AS plural_id
        ORDER BY label, singular_name
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def merge_plural_to_singular(self) -> int:
        """Merge plural entity variants into their singular forms.

        Dynamically finds all plural/singular pairs (including multi-word phrases)
        and transfers relationships from plural to singular before deleting.

        Returns:
            Number of entities merged.
        """
        # First, discover all pairs dynamically
        pairs = await self.find_plural_singular_pairs()

        if not pairs:
            logger.info("No plural/singular pairs found to merge")
            return 0

        logger.info("Found plural/singular pairs to merge", count=len(pairs))

        merged_count = 0

        # Merge each pair individually to handle relationship transfer properly
        for pair in pairs:
            query = """
            MATCH (plural_node)
            WHERE elementId(plural_node) = $plural_id

            MATCH (singular_node)
            WHERE elementId(singular_node) = $singular_id

            // Transfer outgoing relationships
            CALL (singular_node, plural_node) {
                MATCH (plural_node)-[r]->()
                WITH singular_node, r, endNode(r) AS target, type(r) AS rel_type
                CALL apoc.merge.relationship(singular_node, rel_type, {}, properties(r), target, {})
                YIELD rel
                DELETE r
                RETURN count(*) AS out_count
            }

            // Transfer incoming relationships
            CALL (singular_node, plural_node) {
                MATCH ()-[r]->(plural_node)
                WITH singular_node, r, startNode(r) AS source, type(r) AS rel_type
                CALL apoc.merge.relationship(source, rel_type, {}, properties(r), singular_node, {})
                YIELD rel
                DELETE r
                RETURN count(*) AS in_count
            }

            // Delete plural node
            WITH plural_node, singular_node
            DELETE plural_node
            RETURN count(*) AS merged
            """

            async with self.driver.session(database=self.database) as session:
                try:
                    result = await session.run(
                        query,
                        plural_id=pair["plural_id"],
                        singular_id=pair["singular_id"],
                    )
                    record = await result.single()
                    if record and record["merged"] > 0:
                        merged_count += 1
                        logger.debug(
                            "Merged plural to singular",
                            plural=pair["plural_name"],
                            singular=pair["singular_name"],
                            label=pair["label"],
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to merge pair",
                        plural=pair["plural_name"],
                        singular=pair["singular_name"],
                        error=str(e),
                    )

        logger.info("Merged plural entities", count=merged_count)
        return merged_count

    async def run_cleanup(self) -> dict:
        """Run full entity cleanup: delete generics, merge plurals.

        Returns:
            Statistics about the cleanup operation.
        """
        stats = {
            "deleted_generic": 0,
            "merged_plurals": 0,
        }

        # Step 1: Delete generic entities
        stats["deleted_generic"] = await self.delete_generic_entities()

        # Step 2: Merge plural variants into singulars
        stats["merged_plurals"] = await self.merge_plural_to_singular()

        logger.info(
            "Entity cleanup complete",
            deleted=stats["deleted_generic"],
            merged=stats["merged_plurals"],
        )

        return stats
