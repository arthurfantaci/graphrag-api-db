"""Domain-specific extraction prompts for requirements management.

This module provides customized ERExtractionTemplate for the requirements
management domain, including few-shot examples and critical rules for
proper entity classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.pipeline.kg_builder import ERExtractionTemplate


# =============================================================================
# DOMAIN-SPECIFIC INSTRUCTIONS
# =============================================================================

REQUIREMENTS_DOMAIN_INSTRUCTIONS = """
## DOMAIN: Requirements Management and Traceability

You are extracting entities and relationships from technical content about
requirements management, traceability, and systems engineering practices.

## CRITICAL CLASSIFICATION RULES:

### 1. Industry vs Concept Distinction (IMPORTANT!)
- **Industry** = Business sectors/vertical markets ONLY:
  - ✓ aerospace, automotive, medical devices, defense, energy, rail,
    industrial equipment, consumer electronics, telecommunications
- **Concept** = Technologies, methodologies, or technical domains:
  - ✓ AI/ML, IoT, digital transformation, machine learning, agile,
    model-based systems engineering (MBSE), software development
  - ✗ NEVER classify these as Industry

### 2. Name Normalization
- `name` property: Always lowercase, trimmed
  - Example: "Requirements Traceability" → name: "requirements traceability"
- `display_name` property: Preserve original casing for display
  - Example: display_name: "Requirements Traceability"

### 3. Standards Identification
- Always include `organization` property for standards
- Common organizations: ISO, IEC, IEEE, RTCA, FDA, SAE, ECSS
- Examples:
  - ISO 26262 → organization: "ISO"
  - DO-178C → organization: "RTCA"
  - IEC 62304 → organization: "IEC"

### 4. Tool Classification
- Generic tool categories use descriptive names:
  - "requirements management tool", "ALM platform", "version control"
- Specific products include vendor:
  - "jama connect" with vendor: "Jama Software"

## FEW-SHOT EXAMPLES:

### Example 1: Concept with Definition
Text: "Requirements traceability is the ability to trace a requirement
through its entire lifecycle, from origin to implementation and testing."

Extract:
- Entity: Concept
  - name: "requirements traceability"
  - display_name: "Requirements Traceability"
  - definition: "the ability to trace a requirement through its entire
    lifecycle, from origin to implementation and testing"

### Example 2: Standard with Organization
Text: "ISO 26262 defines functional safety requirements for automotive
electrical and electronic systems."

Extract:
- Entity: Standard
  - name: "iso 26262"
  - display_name: "ISO 26262"
  - organization: "ISO"
  - domain: "automotive"
- Entity: Industry
  - name: "automotive"
  - display_name: "Automotive"
  - regulated: true
- Relationship: (iso 26262)-[APPLIES_TO]->(automotive)

### Example 3: Challenge and Solution
Text: "Scope creep remains one of the biggest challenges in product
development. Implementing rigorous change control with traceability
can help address this issue."

Extract:
- Entity: Challenge
  - name: "scope creep"
  - display_name: "Scope Creep"
  - severity: "high"
- Entity: Concept
  - name: "change control"
  - display_name: "Change Control"
- Entity: Concept
  - name: "traceability"
- Relationship: (change control)-[ADDRESSES]->(scope creep)
- Relationship: (traceability)-[ADDRESSES]->(scope creep)

### Example 4: Technology as Concept (NOT Industry!)
Text: "IoT devices in the automotive industry require robust
requirements management to ensure safety."

Extract:
- Entity: Concept (NOT Industry!)
  - name: "iot"
  - display_name: "IoT"
  - aliases: ["internet of things"]
- Entity: Industry
  - name: "automotive"
  - display_name: "Automotive"
- Relationship: (iot)-[APPLIES_TO]->(automotive)

### Example 5: Process Stage and Artifact
Text: "During the requirements gathering phase, the team produces
a requirements specification document (SRS)."

Extract:
- Entity: Processstage
  - name: "requirements gathering"
  - display_name: "Requirements Gathering"
  - sequence: 1
- Entity: Artifact
  - name: "requirements specification"
  - display_name: "Requirements Specification"
  - artifact_type: "specification"
  - abbreviation: "SRS"
- Relationship: (requirements gathering)-[PRODUCES]->(requirements specification)

## COMMON MISTAKES TO AVOID (Negative Examples):

### ✗ WRONG: Concept -[USED_BY]-> Tool
Text: "Teams use Excel and Word for requirements management in software development."

INCORRECT extraction:
- (software development)-[USED_BY]->(excel)  ← WRONG DIRECTION!
- (software development)-[USED_BY]->(word)   ← WRONG DIRECTION!

CORRECT extraction:
- Entity: Tool { name: "excel" }
- Entity: Tool { name: "word" }
- Entity: Concept { name: "requirements management" }
- NO relationship between Concept and Tool (USED_BY is for Role/Industry → Tool)

**Rule**: USED_BY relationships go FROM Role or Industry TO Tool, never from Concept.
- ✓ (systems engineer)-[USED_BY]->(jama connect)
- ✓ (aerospace)-[USED_BY]->(matlab)
- ✗ (automation)-[USED_BY]->(jama connect)

### ✗ WRONG: Certification Organizations as Standards
Text: "TÜV SÜD certified the product for functional safety compliance."

INCORRECT extraction:
- Entity: Standard { name: "tüv süd" }  ← WRONG! TÜV SÜD is an organization

CORRECT extraction:
- Do NOT create an entity for TÜV SÜD (it's a certification body, not a standard)
- Or if relevant, note it as context but not as a Standard node

**Rule**: These are certification ORGANIZATIONS, not standards:
- ✗ TÜV SÜD, TÜV Rheinland, TÜV Nord (German certification bodies)
- ✗ UL (Underwriters Laboratories)
- ✗ SGS, Bureau Veritas, Intertek (testing companies)
- ✗ FDA, FAA, EASA (regulatory agencies - they ENFORCE standards, they ARE NOT standards)

Standards have alphanumeric designations like: ISO 26262, DO-178C, IEC 62304, IEEE 830

### ✗ WRONG: Standard -[APPLIES_TO]-> Concept
Text: "ISO/IEC 12207 covers systems and software engineering lifecycle processes."

INCORRECT extraction:
- (iso/iec 12207)-[APPLIES_TO]->(systems and software engineering)  ← WRONG!

CORRECT extraction:
- Entity: Standard { name: "iso/iec 12207", organization: "ISO/IEC" }
- Entity: Concept { name: "software lifecycle", definition: "..." }
- Relationship: (iso/iec 12207)-[DEFINES]->(software lifecycle)

**Rule**: Standards APPLY_TO Industries (vertical markets), not Concepts.
- ✓ (iso 26262)-[APPLIES_TO]->(automotive)
- ✓ (do-178c)-[APPLIES_TO]->(aerospace)
- ✗ (iso 26262)-[APPLIES_TO]->(functional safety)  ← functional safety is a Concept!

Use DEFINES when a standard defines or specifies a concept.

## ADDITIONAL GUIDELINES:

1. **Be specific**: "bidirectional traceability" is more valuable than just "traceability"
2. **Avoid duplicates**: Don't create separate entities for singular/plural forms
3. **Capture relationships**: The graph structure is as valuable as the entities
4. **Use patterns**: Only create relationships matching the defined PATTERNS
5. **Include context**: Add definition/description properties when text provides them
6. **Check relationship direction**: Always verify source→target matches allowed patterns

"""

# =============================================================================
# TEMPLATE CREATION
# =============================================================================


def create_extraction_template() -> ERExtractionTemplate:
    """Create an ERExtractionTemplate with domain-specific instructions.

    Returns:
        ERExtractionTemplate configured for requirements management domain.

    Example:
        >>> template = create_extraction_template()
        >>> pipeline = SimpleKGPipeline(..., prompt_template=template)
    """
    from neo4j_graphrag.experimental.pipeline.kg_builder import (
        ERExtractionTemplate,
    )

    # Template must include the required JSON output format from neo4j_graphrag
    # plus our domain-specific instructions
    custom_template = (
        """You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Concept", "properties": {{"name": "traceability", "display_name": "Traceability"}} }}],
"relationships": [{{"type": "ADDRESSES", "start_node_id": "0", "end_node_id": "1", "properties": {{}} }}] }}

Use only the following node and relationship types:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

"""
        + REQUIREMENTS_DOMAIN_INSTRUCTIONS
        + """

Input text:

{text}"""
    )

    return ERExtractionTemplate(template=custom_template)


def get_few_shot_examples() -> list[dict]:
    """Get few-shot examples for entity extraction.

    Returns:
        List of example dictionaries with 'text', 'entities', and 'relationships'.
    """
    return [
        {
            "text": (
                "Requirements traceability is critical for regulatory compliance "
                "in the medical device industry, particularly for IEC 62304."
            ),
            "entities": [
                {
                    "type": "Concept",
                    "name": "requirements traceability",
                    "display_name": "Requirements Traceability",
                },
                {
                    "type": "Industry",
                    "name": "medical devices",
                    "display_name": "Medical Devices",
                    "regulated": True,
                },
                {
                    "type": "Standard",
                    "name": "iec 62304",
                    "display_name": "IEC 62304",
                    "organization": "IEC",
                    "domain": "medical",
                },
            ],
            "relationships": [
                {
                    "source": "requirements traceability",
                    "type": "APPLIES_TO",
                    "target": "medical devices",
                },
                {
                    "source": "iec 62304",
                    "type": "APPLIES_TO",
                    "target": "medical devices",
                },
            ],
        },
        {
            "text": (
                "Agile methodologies like Scrum can be challenging to reconcile "
                "with traditional V-Model approaches required by DO-178C in aerospace."
            ),
            "entities": [
                {
                    "type": "Methodology",
                    "name": "agile",
                    "display_name": "Agile",
                    "approach": "iterative",
                },
                {
                    "type": "Methodology",
                    "name": "scrum",
                    "display_name": "Scrum",
                    "approach": "iterative",
                },
                {
                    "type": "Methodology",
                    "name": "v-model",
                    "display_name": "V-Model",
                    "approach": "sequential",
                },
                {
                    "type": "Standard",
                    "name": "do-178c",
                    "display_name": "DO-178C",
                    "organization": "RTCA",
                    "domain": "aerospace",
                },
                {
                    "type": "Industry",
                    "name": "aerospace",
                    "display_name": "Aerospace",
                    "regulated": True,
                },
            ],
            "relationships": [
                {"source": "agile", "type": "ALTERNATIVE_TO", "target": "v-model"},
                {"source": "scrum", "type": "COMPONENT_OF", "target": "agile"},
                {"source": "do-178c", "type": "APPLIES_TO", "target": "aerospace"},
                {"source": "v-model", "type": "APPLIES_TO", "target": "aerospace"},
            ],
        },
    ]
