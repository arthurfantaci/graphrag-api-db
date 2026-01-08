"""LangExtract prompt and example schemas for requirements management domain.

Few-shot examples optimized for extracting entities and relationships
from Jama's Requirements Management Guide. These prompts teach LangExtract
to recognize domain-specific terminology and relationships.

The examples follow LangExtract's ExampleData format with extraction_text
that matches the source verbatim (no paraphrasing).
"""

import textwrap

# =============================================================================
# ENTITY EXTRACTION PROMPT
# =============================================================================

ENTITY_EXTRACTION_PROMPT = textwrap.dedent("""
    Extract domain entities from this requirements management article.

    Entity types to extract:
    - concept: Technical concepts and terminology
      (e.g., "requirements traceability", "verification", "EARS notation")
    - methodology: Development methodologies and frameworks
      (e.g., "Agile", "MBSE", "V-Model", "Waterfall")
    - tool: Software tools and platforms
      (e.g., "Jama Connect", "DOORS", "Excel", "Jira")
    - standard: Industry standards and regulations
      (e.g., "ISO 13485", "DO-178C", "IEC 62304", "FDA 21 CFR Part 820")
    - challenge: Problems and challenges teams face
      (e.g., "scope creep", "requirement ambiguity", "poor traceability")
    - best_practice: Recommended practices and solutions
      (e.g., "live traceability", "baseline management", "requirements reviews")
    - role: Job roles and stakeholders
      (e.g., "product manager", "systems engineer", "QA engineer")
    - industry: Industry verticals
      (e.g., "medical device", "aerospace", "automotive", "defense")
    - artifact: Documents and work products
      (e.g., "SRS", "PRD", "traceability matrix", "DHF")
    - process_stage: Development lifecycle stages
      (e.g., "elicitation", "verification", "validation", "change control")

    IMPORTANT RULES:
    1. Use EXACT text from the input for extraction_text. Do not paraphrase.
    2. Extract entities in order of appearance in the text.
    3. Do not overlap text spans between entities.
    4. Provide meaningful attributes to add context:
       - For concepts: include 'definition' if the text defines it
       - For tools: include 'vendor' if mentioned
       - For standards: include 'organization' (ISO, IEC, FDA, etc.)
       - For challenges: include 'impact' if described
       - For best_practices: include 'benefit' if stated
       - For artifacts: include 'abbreviation' if present
    5. Prefer specific terms over generic ones.
    6. Skip promotional/marketing language about Jama products.
    7. Extract the most canonical form of multi-word terms.
""").strip()


# =============================================================================
# RELATIONSHIP EXTRACTION PROMPT
# =============================================================================

RELATIONSHIP_EXTRACTION_PROMPT = textwrap.dedent("""
    Extract relationships between entities in this requirements management article.

    Relationship types to extract:
    - DEFINES: The text provides a definition for a concept
      (e.g., "Requirements traceability is the ability to...")
    - PREREQUISITE_FOR: Concept A must be understood/done before B
      (e.g., "Before you can verify, you must define requirements")
    - COMPONENT_OF: A is part of B
      (e.g., "verification is part of V&V")
    - ADDRESSES: A practice/tool addresses a challenge
      (e.g., "Live traceability addresses the challenge of outdated links")
    - REQUIRES: A standard/process requires something
      (e.g., "ISO 13485 requires documented design controls")
    - RELATED_TO: General semantic relationship
      (e.g., "traceability is closely related to change management")
    - ALTERNATIVE_TO: Comparing alternatives
      (e.g., "live traceability vs after-the-fact traceability")
    - USED_BY: Tool/practice used by an industry or role
      (e.g., "MBSE is used by aerospace companies")
    - APPLIES_TO: Standard applies to an industry
      (e.g., "DO-178C applies to aerospace software")
    - PRODUCES: Process produces an artifact
      (e.g., "requirements elicitation produces the SRS")

    IMPORTANT RULES:
    1. Both source and target entities must appear in the text.
    2. Extract the evidence text (the passage that supports the relationship).
    3. Relationships should be directional and specific.
    4. Avoid trivial relationships (mere co-occurrence is not a relationship).
    5. Focus on relationships that would help answer "how" and "why" questions.
    6. Each relationship should have clear evidence in the text.
""").strip()


# =============================================================================
# SUMMARY GENERATION PROMPT
# =============================================================================

SUMMARY_PROMPT = textwrap.dedent("""
    Generate a concise 2-3 sentence summary of this requirements management article.

    Focus on:
    1. The main topic or concept being explained
    2. Key takeaways or recommendations
    3. Who this information is most relevant for (roles, industries)

    Guidelines:
    - Keep it factual and avoid promotional language
    - Use domain-appropriate terminology
    - Make it useful for someone searching for this topic
    - Do not start with "This article..."
""").strip()


# =============================================================================
# GLOSSARY ENRICHMENT PROMPT
# =============================================================================

GLOSSARY_ENRICHMENT_PROMPT = textwrap.dedent("""
    Given this glossary term and its definition from a requirements management guide,
    identify related concepts and relevant chapters.

    Term: {term}
    Definition: {definition}

    Chapter topics for reference:
    1. Requirements Management
    2. Verification and Validation
    3. Requirements Definition
    4. Best Practices
    5. Traceability
    6. Change Management
    7. Requirements Analysis
    8. Requirements Tools
    9. Requirements Engineering
    10. V-Model
    11. Regulatory Compliance
    12. Industry Applications
    13. Implementation
    14. Documentation
    15. Quality Assurance

    Respond with JSON:
    {{
        "related_terms": ["term1", "term2"],
        "relevant_chapter_numbers": [1, 5, 11]
    }}
""").strip()


# =============================================================================
# FEW-SHOT EXAMPLES FOR ENTITY EXTRACTION
# =============================================================================


def get_entity_examples() -> list:
    """Get few-shot examples for entity extraction.

    Returns LangExtract ExampleData objects demonstrating proper
    entity extraction from requirements management text.

    Note: Import langextract only when this function is called to avoid
    import errors when the package isn't installed.
    """
    import langextract as lx

    return [
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Requirements traceability is the ability to trace requirements
                throughout the product lifecycle, from origin through development
                to deployment. Live Traceability, a feature of Jama Connect, enables
                real-time visibility into requirement relationships and helps teams
                maintain compliance with ISO 13485 for medical devices.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="Requirements traceability",
                    attributes={
                        "definition": (
                            "the ability to trace requirements throughout "
                            "the product lifecycle"
                        ),
                        "scope": "from origin through development to deployment",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="best_practice",
                    extraction_text="Live Traceability",
                    attributes={
                        "benefit": "real-time visibility into relationships",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="standard",
                    extraction_text="ISO 13485",
                    attributes={
                        "organization": "ISO",
                        "domain": "medical devices",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="industry",
                    extraction_text="medical devices",
                    attributes={},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                The product manager is responsible for gathering stakeholder
                requirements and ensuring the SRS (System Requirements Specification)
                accurately captures all functional and non-functional requirements.
                Common challenges include scope creep and ambiguous requirements,
                which can lead to costly rework during development.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="role",
                    extraction_text="product manager",
                    attributes={
                        "responsibility": "gathering stakeholder requirements",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="artifact",
                    extraction_text="SRS (System Requirements Specification)",
                    attributes={
                        "abbreviation": "SRS",
                        "purpose": "captures functional and non-functional reqs",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="functional and non-functional requirements",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="challenge",
                    extraction_text="scope creep",
                    attributes={
                        "impact": "costly rework during development",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="challenge",
                    extraction_text="ambiguous requirements",
                    attributes={
                        "impact": "costly rework during development",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Model-Based Systems Engineering (MBSE) replaces document-centric
                approaches with a data-centric methodology. Systems engineers use
                SysML diagrams to model system architecture, improving communication
                and reducing errors in aerospace and defense projects following DO-178C.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="methodology",
                    extraction_text="Model-Based Systems Engineering (MBSE)",
                    attributes={
                        "abbreviation": "MBSE",
                        "approach": "data-centric",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="role",
                    extraction_text="Systems engineers",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="artifact",
                    extraction_text="SysML diagrams",
                    attributes={
                        "purpose": "model system architecture",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="industry",
                    extraction_text="aerospace and defense",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="standard",
                    extraction_text="DO-178C",
                    attributes={
                        "domain": "aerospace software",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                The V-Model emphasizes that verification and validation activities
                should be planned in parallel with development phases. During the
                requirements elicitation phase, teams should also plan acceptance
                testing criteria to ensure requirements are testable.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="methodology",
                    extraction_text="V-Model",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="verification and validation",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="process_stage",
                    extraction_text="requirements elicitation",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="process_stage",
                    extraction_text="acceptance testing",
                    attributes={},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                The EARS (Easy Approach to Requirements Syntax) notation provides
                templates for writing clear, unambiguous requirements. Using EARS
                helps prevent the challenge of requirement ambiguity by enforcing
                consistent structure across all system requirements.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="methodology",
                    extraction_text="EARS (Easy Approach to Requirements Syntax)",
                    attributes={
                        "abbreviation": "EARS",
                        "purpose": "templates for clear, unambiguous requirements",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="challenge",
                    extraction_text="requirement ambiguity",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="artifact",
                    extraction_text="system requirements",
                    attributes={},
                ),
            ],
        ),
    ]


# =============================================================================
# FEW-SHOT EXAMPLES FOR RELATIONSHIP EXTRACTION
# =============================================================================


def get_relationship_examples() -> list:
    """Get few-shot examples for relationship extraction.

    Returns LangExtract ExampleData objects demonstrating proper
    relationship extraction between entities.
    """
    import langextract as lx

    return [
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Requirements traceability is the ability to trace requirements
                throughout the product lifecycle. Live Traceability addresses
                the challenge of maintaining up-to-date requirement relationships,
                which is critical for compliance with ISO 13485 in medical device
                development.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="DEFINES",
                    extraction_text=(
                        "Requirements traceability is the ability to trace "
                        "requirements throughout the product lifecycle"
                    ),
                    attributes={
                        "source": "Requirements traceability",
                        "target": (
                            "ability to trace requirements throughout "
                            "the product lifecycle"
                        ),
                    },
                ),
                lx.data.Extraction(
                    extraction_class="ADDRESSES",
                    extraction_text=(
                        "Live Traceability addresses the challenge of "
                        "maintaining up-to-date requirement relationships"
                    ),
                    attributes={
                        "source": "Live Traceability",
                        "target": "maintaining up-to-date requirement relationships",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="APPLIES_TO",
                    extraction_text="ISO 13485 in medical device development",
                    attributes={
                        "source": "ISO 13485",
                        "target": "medical device development",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Before beginning verification activities, the team must complete
                requirements definition. Verification is a component of the broader
                V&V (Verification and Validation) process.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="PREREQUISITE_FOR",
                    extraction_text=(
                        "Before beginning verification activities, the team "
                        "must complete requirements definition"
                    ),
                    attributes={
                        "source": "requirements definition",
                        "target": "verification",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="COMPONENT_OF",
                    extraction_text=(
                        "Verification is a component of the broader V&V "
                        "(Verification and Validation) process"
                    ),
                    attributes={
                        "source": "Verification",
                        "target": "V&V",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Aerospace companies following DO-178C require rigorous requirements
                traceability. The requirements elicitation process produces the
                System Requirements Specification document.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="REQUIRES",
                    extraction_text=(
                        "Aerospace companies following DO-178C require "
                        "rigorous requirements traceability"
                    ),
                    attributes={
                        "source": "DO-178C",
                        "target": "requirements traceability",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="PRODUCES",
                    extraction_text=(
                        "requirements elicitation process produces the "
                        "System Requirements Specification"
                    ),
                    attributes={
                        "source": "requirements elicitation",
                        "target": "System Requirements Specification",
                    },
                ),
            ],
        ),
    ]


# =============================================================================
# FEW-SHOT EXAMPLES FOR SUMMARY GENERATION
# =============================================================================


def get_summary_examples() -> list:
    """Get few-shot examples for summary generation.

    Returns LangExtract ExampleData objects demonstrating proper
    article summary generation.
    """
    import langextract as lx

    return [
        lx.data.ExampleData(
            text=textwrap.dedent("""
                Requirements traceability is the ability to trace requirements
                throughout the product lifecycle, from origin through development
                to deployment. It enables teams to understand the impact of changes,
                verify that all requirements are implemented, and demonstrate
                compliance with regulatory standards. Organizations in regulated
                industries like medical devices and aerospace particularly benefit
                from robust traceability practices.
            """).strip(),
            extractions=[
                lx.data.Extraction(
                    extraction_class="summary",
                    extraction_text=(
                        "Requirements traceability enables tracking requirements "
                        "from origin through deployment, helping teams manage "
                        "changes and verify implementation completeness. Essential "
                        "for regulated industries like medical devices and "
                        "aerospace where compliance documentation is mandatory."
                    ),
                    attributes={},
                ),
            ],
        ),
    ]
