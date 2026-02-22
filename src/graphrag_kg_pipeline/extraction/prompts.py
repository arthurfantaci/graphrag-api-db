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
- Entity: Tool (name: "excel")
- Entity: Tool (name: "word")
- Entity: Concept (name: "requirements management")
- NO relationship between Concept and Tool (USED_BY is for Role/Industry → Tool)

**Rule**: USED_BY relationships go FROM Role or Industry TO Tool, never from Concept.
- ✓ (systems engineer)-[USED_BY]->(jama connect)
- ✓ (aerospace)-[USED_BY]->(matlab)
- ✗ (automation)-[USED_BY]->(jama connect)

### Organization vs Standard Distinction (IMPORTANT!)
Text: "TÜV SÜD certified the product for functional safety compliance."

INCORRECT extraction:
- Entity: Standard (name: "tüv süd")  ← WRONG! TÜV SÜD is an organization

CORRECT extraction:
- Entity: Organization (name: "tüv süd", organization_type: "certification_body")

**Rule**: These are ORGANIZATIONS, not Standards — extract them as Organization:
- ✓ TÜV SÜD, TÜV Rheinland, TÜV Nord → Organization (certification_body)
- ✓ UL (Underwriters Laboratories) → Organization (certification_body)
- ✓ SGS, Bureau Veritas, Intertek → Organization (certification_body)
- ✓ FDA, FAA, EASA → Organization (regulatory_agency)
- ✓ ISO, IEC, IEEE, RTCA, SAE → Organization (standards_body)
- ✓ Jama Software, Siemens, IBM → Organization (vendor)
- ✓ NASA, INCOSE, PMI → Organization (professional_society)

Standards have alphanumeric designations like: ISO 26262, DO-178C, IEC 62304, IEEE 830

### Example: Organization PUBLISHES Standard
Text: "RTCA publishes DO-178C, the primary safety standard for airborne software."

Extract:
- Entity: Organization (name: "rtca", organization_type: "standards_body")
- Entity: Standard (name: "do-178c", organization: "RTCA")
- Relationship: (rtca)-[PUBLISHES]->(do-178c)

### ✗ WRONG: Standard -[APPLIES_TO]-> Concept
Text: "ISO/IEC 12207 covers systems and software engineering lifecycle processes."

INCORRECT extraction:
- (iso/iec 12207)-[APPLIES_TO]->(systems and software engineering)  ← WRONG!

CORRECT extraction:
- Entity: Standard (name: "iso/iec 12207", organization: "ISO/IEC")
- Entity: Concept (name: "software lifecycle", definition: "...")
- Relationship: (iso/iec 12207)-[DEFINES]->(software lifecycle)

**Rule**: Standards APPLY_TO Industries (vertical markets), not Concepts.
- ✓ (iso 26262)-[APPLIES_TO]->(automotive)
- ✓ (do-178c)-[APPLIES_TO]->(aerospace)
- ✗ (iso 26262)-[APPLIES_TO]->(functional safety)  ← functional safety is a Concept!

Use DEFINES when a standard defines or specifies a concept.

### 5. Challenge vs Outcome Classification (IMPORTANT!)
- **Challenge** = ONLY obstacles, difficulties, problems, risks that need to be overcome
  - ✓ scope creep, requirements volatility, incomplete traceability, late defect discovery
- **Outcome** = Positive results, benefits, goals achieved through good practices
  - ✓ improved product quality, reduced time-to-market, regulatory compliance
  - ✓ customer satisfaction, cost reduction, faster development cycles
- **NEVER classify positive outcomes as Challenge:**
  - ✗ "High-Quality Products" → NOT a Challenge → use Outcome
  - ✗ "Customer Satisfaction" → NOT a Challenge → use Outcome
  - ✗ "Reduced Time-to-Market" → NOT a Challenge → use Outcome

### 6. Definition Extraction (CRITICAL!)
- ALWAYS include definitions: When the text contains a definition, explanation,
  or description of a concept, you MUST include it as the `definition` property.
  This is critical for knowledge graph completeness.
- Even partial definitions are valuable — capture them.
- If a term is defined with "is", "refers to", "means", or "involves", extract the definition.

## ADDITIONAL GUIDELINES:

1. **Be specific**: "bidirectional traceability" is more valuable than just "traceability"
2. **Avoid duplicates**: Don't create separate entities for singular/plural forms
3. **Capture relationships**: The graph structure is as valuable as the entities
4. **Use patterns**: Only create relationships matching the defined PATTERNS
5. **Check relationship direction**: Always verify source→target matches allowed patterns
6. **Extract all standards and industries**: Even when mentioned in passing, extract Standard
   and Industry entities. The MENTIONED_IN relationship to their source chunk is valuable.

## NORMALIZATION RULES (CRITICAL):

### 1. Always Use SINGULAR Form
- ✓ "requirement" (not "requirements")
- ✓ "stakeholder" (not "stakeholders")
- ✓ "artifact" (not "artifacts")
- ✓ "constraint" (not "constraints")
- ✓ "specification" (not "specifications")

### 2. NEVER Extract These Generic Terms as Entities
The following terms are too vague to be useful - DO NOT create entities for them:
- Generic tools: "tool", "tools", "software", "solution", "platform", "system"
- Generic processes: "method", "process", "approach", "technique", "procedure"
- Generic documents: "document", "file", "report"
- Generic people: "person", "user", "team", "member"
- Generic abstractions: "thing", "item", "element", "component", "part", "type"

### 3. Prefer Specific Over Generic
- ✓ "requirements management tool" over "tool"
- ✓ "traceability matrix" over "document"
- ✓ "systems engineer" over "user"
- ✓ "agile methodology" over "process"

If a term is too generic to provide meaningful information, skip it entirely.

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

    Returns a pool of curated examples covering all 10 entity types,
    common error cases, and correct definition extraction.

    Returns:
        List of example dictionaries with 'text', 'entities', and 'relationships'.
    """
    return [
        # Example 1: Concept + Standard + Industry (with definition)
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
        # Example 2: Methodology + Standard + Industry
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
        # Example 3: Challenge + Bestpractice + definition extraction
        {
            "text": (
                "Scope creep refers to the uncontrolled expansion of project scope "
                "without adjustments to time, cost, or resources. A best practice to "
                "address scope creep is implementing a formal change control process."
            ),
            "entities": [
                {
                    "type": "Challenge",
                    "name": "scope creep",
                    "display_name": "Scope Creep",
                    "definition": (
                        "the uncontrolled expansion of project scope "
                        "without adjustments to time, cost, or resources"
                    ),
                    "severity": "high",
                },
                {
                    "type": "Bestpractice",
                    "name": "formal change control process",
                    "display_name": "Formal Change Control Process",
                    "rationale": "addresses scope creep by controlling changes",
                },
            ],
            "relationships": [
                {
                    "source": "formal change control process",
                    "type": "ADDRESSES",
                    "target": "scope creep",
                },
            ],
        },
        # Example 4: Tool + Role + Artifact
        {
            "text": (
                "Jama Connect enables requirements engineers to create and maintain "
                "a requirements traceability matrix (RTM) throughout the lifecycle."
            ),
            "entities": [
                {
                    "type": "Tool",
                    "name": "jama connect",
                    "display_name": "Jama Connect",
                    "vendor": "Jama Software",
                    "category": "requirements management",
                },
                {
                    "type": "Role",
                    "name": "requirements engineer",
                    "display_name": "Requirements Engineer",
                },
                {
                    "type": "Artifact",
                    "name": "requirements traceability matrix",
                    "display_name": "Requirements Traceability Matrix",
                    "abbreviation": "RTM",
                    "artifact_type": "matrix",
                },
            ],
            "relationships": [
                {
                    "source": "requirements engineer",
                    "type": "USED_BY",
                    "target": "jama connect",
                },
                {
                    "source": "requirements engineer",
                    "type": "PRODUCES",
                    "target": "requirements traceability matrix",
                },
            ],
        },
        # Example 5: Outcome (positive result, NOT a Challenge)
        {
            "text": (
                "Achieving high-quality products requires establishing strong "
                "requirements management practices early in the development lifecycle."
            ),
            "entities": [
                {
                    "type": "Outcome",
                    "name": "high-quality products",
                    "display_name": "High-Quality Products",
                    "outcome_type": "quality",
                },
                {
                    "type": "Concept",
                    "name": "requirements management",
                    "display_name": "Requirements Management",
                    "definition": (
                        "practices for managing requirements throughout the development lifecycle"
                    ),
                },
            ],
            "relationships": [
                {
                    "source": "requirements management",
                    "type": "ACHIEVES",
                    "target": "high-quality products",
                },
            ],
        },
        # Example 6: ProcessStage + PREREQUISITE_FOR
        {
            "text": (
                "Requirements validation must be completed before the design "
                "phase begins. During validation, business analysts verify that "
                "requirements accurately reflect stakeholder needs."
            ),
            "entities": [
                {
                    "type": "Processstage",
                    "name": "requirements validation",
                    "display_name": "Requirements Validation",
                    "definition": (
                        "verifying that requirements accurately reflect stakeholder needs"
                    ),
                },
                {
                    "type": "Processstage",
                    "name": "design phase",
                    "display_name": "Design Phase",
                },
                {
                    "type": "Role",
                    "name": "business analyst",
                    "display_name": "Business Analyst",
                },
            ],
            "relationships": [
                {
                    "source": "requirements validation",
                    "type": "PREREQUISITE_FOR",
                    "target": "design phase",
                },
            ],
        },
        # Example 7: Standard DEFINES concept (not APPLIES_TO)
        {
            "text": (
                "ISO 26262 defines functional safety levels (ASIL A through D) "
                "for automotive electronic systems."
            ),
            "entities": [
                {
                    "type": "Standard",
                    "name": "iso 26262",
                    "display_name": "ISO 26262",
                    "organization": "ISO",
                    "domain": "automotive",
                },
                {
                    "type": "Concept",
                    "name": "functional safety level",
                    "display_name": "Functional Safety Level",
                    "definition": "safety integrity levels from ASIL A through D",
                    "aliases": ["ASIL"],
                },
                {
                    "type": "Industry",
                    "name": "automotive",
                    "display_name": "Automotive",
                    "regulated": True,
                },
            ],
            "relationships": [
                {
                    "source": "iso 26262",
                    "type": "DEFINES",
                    "target": "functional safety level",
                },
                {"source": "iso 26262", "type": "APPLIES_TO", "target": "automotive"},
            ],
        },
        # Example 8: Organization + PUBLISHES + REGULATES
        {
            "text": (
                "The FDA regulates the medical device industry and enforces "
                "compliance with IEC 62304 for software lifecycle processes."
            ),
            "entities": [
                {
                    "type": "Organization",
                    "name": "fda",
                    "display_name": "FDA",
                    "organization_type": "regulatory_agency",
                    "abbreviation": "FDA",
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
                {"source": "fda", "type": "REGULATES", "target": "medical devices"},
                {"source": "iec 62304", "type": "APPLIES_TO", "target": "medical devices"},
            ],
        },
    ]
