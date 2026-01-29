"""Knowledge graph schema for requirements management domain.

This module defines the complete schema for entity and relationship extraction:
- 10 node types (Concept, Challenge, Artifact, etc.)
- 10 relationship types (ADDRESSES, REQUIRES, etc.)
- ~30 patterns constraining valid (source, rel, target) triples

The schema is designed for neo4j_graphrag's SimpleKGPipeline.
"""

from typing import Any

# =============================================================================
# NODE TYPE DEFINITIONS
# =============================================================================

NODE_TYPES: dict[str, dict[str, Any]] = {
    "Concept": {
        "label": "Concept",
        "description": (
            "A requirements management concept, principle, or technique. "
            "Examples: traceability, requirements elicitation, impact analysis, "
            "bidirectional traceability, verification, validation. "
            "NOTE: Technical concepts like AI/ML, IoT, digital transformation "
            "are Concepts, NOT Industries."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name (e.g., 'requirements traceability')",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "definition": {
                "type": "STRING",
                "required": False,
                "description": "Brief definition if available",
            },
            "aliases": {
                "type": "LIST",
                "required": False,
                "description": "Alternative names or abbreviations",
            },
        },
    },
    "Challenge": {
        "label": "Challenge",
        "description": (
            "A problem, risk, or challenge in requirements management. "
            "Examples: scope creep, requirements volatility, incomplete traceability, "
            "stakeholder miscommunication, late defect discovery."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "severity": {
                "type": "STRING",
                "required": False,
                "description": "Impact level: high, medium, low",
            },
        },
    },
    "Artifact": {
        "label": "Artifact",
        "description": (
            "A document, deliverable, or work product. "
            "Examples: requirements specification, RTM (requirements traceability matrix), "
            "test plan, design document, change request, user story."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "artifact_type": {
                "type": "STRING",
                "required": False,
                "description": "Classification: document, matrix, plan, specification",
            },
            "abbreviation": {
                "type": "STRING",
                "required": False,
                "description": "Common abbreviation (e.g., 'RTM', 'SRS')",
            },
        },
    },
    "Bestpractice": {
        "label": "Bestpractice",
        "description": (
            "A recommended practice, guideline, or approach. "
            "Examples: maintain bidirectional links, involve stakeholders early, "
            "use version control, conduct impact analysis before changes."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "rationale": {
                "type": "STRING",
                "required": False,
                "description": "Why this practice is recommended",
            },
        },
    },
    "Processstage": {
        "label": "Processstage",
        "description": (
            "A phase or stage in a development lifecycle. "
            "Examples: requirements gathering, design phase, implementation, "
            "verification, validation, deployment, maintenance."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "sequence": {
                "type": "INTEGER",
                "required": False,
                "description": "Typical order in lifecycle (1, 2, 3, ...)",
            },
        },
    },
    "Role": {
        "label": "Role",
        "description": (
            "A job role or stakeholder type involved in requirements. "
            "Examples: requirements engineer, product owner, business analyst, "
            "systems engineer, quality assurance, project manager."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Original casing for display",
            },
            "responsibilities": {
                "type": "STRING",
                "required": False,
                "description": "Primary responsibilities",
            },
        },
    },
    "Standard": {
        "label": "Standard",
        "description": (
            "An industry standard, regulation, or compliance framework. "
            "Examples: ISO 26262, DO-178C, IEC 62304, FDA 21 CFR Part 11, "
            "CMMI, ASPICE, MIL-STD-498."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Official name with proper casing",
            },
            "organization": {
                "type": "STRING",
                "required": False,
                "description": "Issuing organization (ISO, RTCA, FDA, etc.)",
            },
            "domain": {
                "type": "STRING",
                "required": False,
                "description": "Application domain (automotive, aerospace, medical)",
            },
        },
    },
    "Tool": {
        "label": "Tool",
        "description": (
            "A software tool or tool category for requirements management. "
            "Examples: Jama Connect, requirements management tool, ALM platform, "
            "version control system, collaboration tool."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Product name with proper casing",
            },
            "vendor": {
                "type": "STRING",
                "required": False,
                "description": "Tool vendor if specific product",
            },
            "category": {
                "type": "STRING",
                "required": False,
                "description": "Tool category (RM tool, ALM, testing, etc.)",
            },
        },
    },
    "Methodology": {
        "label": "Methodology",
        "description": (
            "A development methodology or framework. "
            "Examples: Agile, Scrum, Waterfall, V-Model, SAFe, "
            "Model-Based Systems Engineering (MBSE), DevOps."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase normalized name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Framework name with proper casing",
            },
            "approach": {
                "type": "STRING",
                "required": False,
                "description": "Iterative, sequential, hybrid",
            },
        },
    },
    "Industry": {
        "label": "Industry",
        "description": (
            "A business sector or vertical market. "
            "ONLY use for actual industries like: aerospace, automotive, "
            "medical devices, defense, industrial equipment, energy. "
            "NOTE: AI/ML, IoT, software, digital transformation are "
            "Concepts, NOT Industries."
        ),
        "properties": {
            "name": {
                "type": "STRING",
                "required": True,
                "description": "Lowercase canonical industry name",
            },
            "display_name": {
                "type": "STRING",
                "required": False,
                "description": "Display name with proper casing",
            },
            "regulated": {
                "type": "BOOLEAN",
                "required": False,
                "description": "Whether this industry is heavily regulated",
            },
        },
    },
}


# =============================================================================
# RELATIONSHIP TYPE DEFINITIONS
# =============================================================================

RELATIONSHIP_TYPES: dict[str, dict[str, Any]] = {
    "ADDRESSES": {
        "type": "ADDRESSES",
        "description": (
            "A concept, best practice, or tool addresses/solves a challenge. "
            "Example: (traceability)-[ADDRESSES]->(scope creep)"
        ),
        "properties": {
            "effectiveness": {
                "type": "STRING",
                "required": False,
                "description": "How effective: high, medium, low",
            },
        },
    },
    "REQUIRES": {
        "type": "REQUIRES",
        "description": (
            "One concept or artifact requires another. "
            "Example: (impact analysis)-[REQUIRES]->(traceability)"
        ),
        "properties": {},
    },
    "COMPONENT_OF": {
        "type": "COMPONENT_OF",
        "description": (
            "One artifact or concept is a component/part of another. "
            "Example: (requirement)-[COMPONENT_OF]->(requirements specification)"
        ),
        "properties": {},
    },
    "RELATED_TO": {
        "type": "RELATED_TO",
        "description": (
            "General semantic relationship between concepts. "
            "Use when more specific relationships don't apply."
        ),
        "properties": {
            "relationship_nature": {
                "type": "STRING",
                "required": False,
                "description": "Nature of the relationship",
            },
        },
    },
    "ALTERNATIVE_TO": {
        "type": "ALTERNATIVE_TO",
        "description": (
            "Two items are alternatives or competing approaches. "
            "Example: (agile)-[ALTERNATIVE_TO]->(waterfall)"
        ),
        "properties": {},
    },
    "USED_BY": {
        "type": "USED_BY",
        "description": (
            "A tool, artifact, or concept is used by a role or industry. "
            "Example: (requirements engineer)-[USED_BY]->(RTM)"
        ),
        "properties": {},
    },
    "APPLIES_TO": {
        "type": "APPLIES_TO",
        "description": (
            "A standard, methodology, or practice applies to an industry or domain. "
            "Example: (ISO 26262)-[APPLIES_TO]->(automotive)"
        ),
        "properties": {},
    },
    "PRODUCES": {
        "type": "PRODUCES",
        "description": (
            "A process stage or role produces an artifact. "
            "Example: (requirements gathering)-[PRODUCES]->(requirements specification)"
        ),
        "properties": {},
    },
    "DEFINES": {
        "type": "DEFINES",
        "description": (
            "A standard or artifact defines requirements for something. "
            "Example: (DO-178C)-[DEFINES]->(software levels)"
        ),
        "properties": {},
    },
    "PREREQUISITE_FOR": {
        "type": "PREREQUISITE_FOR",
        "description": (
            "One item must be completed before another can begin. "
            "Example: (requirements gathering)-[PREREQUISITE_FOR]->(design phase)"
        ),
        "properties": {},
    },
}


# =============================================================================
# VALID PATTERNS (Source Type, Relationship, Target Type)
# =============================================================================

PATTERNS: list[tuple[str, str, str]] = [
    # ADDRESSES patterns
    ("Concept", "ADDRESSES", "Challenge"),
    ("Bestpractice", "ADDRESSES", "Challenge"),
    ("Tool", "ADDRESSES", "Challenge"),
    ("Methodology", "ADDRESSES", "Challenge"),
    # REQUIRES patterns
    ("Concept", "REQUIRES", "Concept"),
    ("Concept", "REQUIRES", "Artifact"),
    ("Bestpractice", "REQUIRES", "Concept"),
    ("Processstage", "REQUIRES", "Artifact"),
    ("Tool", "REQUIRES", "Concept"),
    # COMPONENT_OF patterns
    ("Artifact", "COMPONENT_OF", "Artifact"),
    ("Concept", "COMPONENT_OF", "Concept"),
    ("Processstage", "COMPONENT_OF", "Methodology"),
    # RELATED_TO patterns (broader applicability)
    ("Concept", "RELATED_TO", "Concept"),
    ("Challenge", "RELATED_TO", "Challenge"),
    ("Artifact", "RELATED_TO", "Artifact"),
    ("Standard", "RELATED_TO", "Standard"),
    ("Bestpractice", "RELATED_TO", "Concept"),
    # ALTERNATIVE_TO patterns
    ("Methodology", "ALTERNATIVE_TO", "Methodology"),
    ("Tool", "ALTERNATIVE_TO", "Tool"),
    ("Concept", "ALTERNATIVE_TO", "Concept"),
    # USED_BY patterns
    ("Role", "USED_BY", "Tool"),
    ("Role", "USED_BY", "Artifact"),
    ("Industry", "USED_BY", "Tool"),
    ("Industry", "USED_BY", "Methodology"),
    # APPLIES_TO patterns
    ("Standard", "APPLIES_TO", "Industry"),
    ("Methodology", "APPLIES_TO", "Industry"),
    ("Bestpractice", "APPLIES_TO", "Processstage"),
    ("Concept", "APPLIES_TO", "Processstage"),
    # PRODUCES patterns
    ("Processstage", "PRODUCES", "Artifact"),
    ("Role", "PRODUCES", "Artifact"),
    # DEFINES patterns
    ("Standard", "DEFINES", "Concept"),
    ("Standard", "DEFINES", "Artifact"),
    # PREREQUISITE_FOR patterns
    ("Processstage", "PREREQUISITE_FOR", "Processstage"),
    ("Artifact", "PREREQUISITE_FOR", "Processstage"),
    ("Concept", "PREREQUISITE_FOR", "Concept"),
]


def get_schema_for_pipeline() -> dict[str, Any]:
    """Get schema formatted for neo4j_graphrag SimpleKGPipeline.

    Returns:
        Dictionary with 'node_types', 'relationship_types', and 'patterns' keys
        in the format expected by neo4j_graphrag's schema parameter.

    Example:
        >>> schema = get_schema_for_pipeline()
        >>> pipeline = SimpleKGPipeline(..., schema=schema)
    """
    # Format node_types for neo4j_graphrag
    # Can be simple strings or dicts with label, description, properties
    node_types = []
    for label, config in NODE_TYPES.items():
        node_def = {
            "label": label,
            "description": config["description"],
            "properties": [
                {
                    "name": prop_name,
                    "type": prop_config["type"],
                }
                for prop_name, prop_config in config["properties"].items()
            ],
        }
        node_types.append(node_def)

    # Format relationship_types for neo4j_graphrag
    relationship_types = []
    for rel_type, config in RELATIONSHIP_TYPES.items():
        rel_props = config.get("properties", {})
        if rel_props:
            rel_def = {
                "label": rel_type,
                "description": config["description"],
                "properties": [
                    {
                        "name": prop_name,
                        "type": prop_config["type"],
                    }
                    for prop_name, prop_config in rel_props.items()
                ],
            }
        else:
            # Simple relationship with just label and description
            rel_def = {
                "label": rel_type,
                "description": config["description"],
            }
        relationship_types.append(rel_def)

    return {
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": PATTERNS,
    }


def get_node_type_names() -> list[str]:
    """Get list of all node type labels.

    Returns:
        List of node type labels.
    """
    return list(NODE_TYPES.keys())


def get_relationship_type_names() -> list[str]:
    """Get list of all relationship type names.

    Returns:
        List of relationship type names.
    """
    return list(RELATIONSHIP_TYPES.keys())


def validate_pattern(source_type: str, rel_type: str, target_type: str) -> bool:
    """Check if a (source, relationship, target) pattern is valid.

    Args:
        source_type: Source node type label.
        rel_type: Relationship type.
        target_type: Target node type label.

    Returns:
        True if this pattern is allowed by the schema.
    """
    return (source_type, rel_type, target_type) in PATTERNS
