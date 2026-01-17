# Sub-Agent Analysis Report: LangChain HTML Processing & neo4j_graphrag Integration

## Executive Summary

This analysis evaluates LangChain's HTML processing capabilities and patterns from the `genai-graphrag-python` examples repository to determine enhancements for the Jama Guide ETL pipeline. **The findings strongly support integrating LangChain's specialized HTML splitters**, which provide structure-aware chunking that preserves semantic hierarchy—a significant improvement over generic text splitting for HTML content.

---

## 1. LangChain HTML Splitters Analysis

### 1.1 Available HTML Splitters

LangChain provides **THREE specialized HTML splitters**, each with distinct use cases:

| Splitter | Use Case | Key Feature |
|----------|----------|-------------|
| `HTMLHeaderTextSplitter` | Header-based hierarchy | Tracks header metadata (h1→h2→h3) |
| `HTMLSectionSplitter` | Larger sections | XSLT transformations, font-size detection |
| `HTMLSemanticPreservingSplitter` | Complex content | Preserves tables, lists intact |

### 1.2 HTMLHeaderTextSplitter (RECOMMENDED for Jama Guide)

**Why this is ideal for Jama content:**
- Jama articles use clear header hierarchy (h1 for title, h2/h3 for sections)
- Automatically tracks which headers apply to each chunk via metadata
- Can be combined with RecursiveCharacterTextSplitter for large sections

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "article_title"),
    ("h2", "section"),
    ("h3", "subsection"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False  # Combine elements under same header
)

# Result: Documents with metadata like:
# {"article_title": "Requirements Traceability", "section": "Benefits"}
```

### 1.3 HTMLSectionSplitter

**Unique capabilities:**
- Uses XSLT transformations to convert custom tags to headers
- Detects sections based on font size (useful for PDFs converted to HTML)
- Can process Jama's Avia theme elements (definition boxes, callouts)

```python
from langchain_text_splitters import HTMLSectionSplitter

# Can use custom XSLT to convert Jama's .avia_message_box to headers
html_splitter = HTMLSectionSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
    xslt_path="./jama_custom_transform.xslt"  # Custom transformation
)
```

### 1.4 HTMLSemanticPreservingSplitter (NEW - Best for Complex Content)

**Critical advantage:** Prevents splitting tables and lists mid-element.

```python
from langchain_text_splitters import HTMLSemanticPreservingSplitter

splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=["h1", "h2", "h3"],
    max_chunk_size=1500,
    preserve_elements=["table", "ul", "ol", "figure"],  # Keep these intact
    custom_handlers={
        "div.avia_message_box": "preserve",  # Jama definition boxes
        "figure": "preserve",
    }
)
```

**⚠️ Important:** `max_chunk_size` is not a hard limit—preserved elements may exceed it.

---

## 2. genai-graphrag-python Examples Analysis

### 2.1 Key Patterns Discovered

| Example File | Pattern | Applicability to Jama ETL |
|--------------|---------|---------------------------|
| `text_splitter_langchain.py` | `LangChainTextSplitterAdapter` | **HIGH** - Integrate LangChain splitters with neo4j_graphrag |
| `text_splitter_section.py` | Custom `TextSplitter` subclass | **HIGH** - Hierarchical chunking implementation |
| `entity_extraction_prompt.py` | `ERExtractionTemplate` customization | **HIGH** - Domain-specific extraction |
| `data_loader_wikipedia.py` | Custom `DataLoader` for web content | **HIGH** - HTML article loading |
| `lexical_graph_config.py` | `LexicalGraphConfig` customization | **MEDIUM** - Custom node labels |
| `extract_entities.py` | Standalone `LLMEntityRelationExtractor` | **MEDIUM** - Testing extraction quality |
| `no_entity_resolution.py` | `perform_entity_resolution=False` | **LOW** - We want resolution enabled |
| `data_loader_custom_pdf.py` | Custom preprocessing in loader | **HIGH** - HTML cleanup pattern |

### 2.2 LangChainTextSplitterAdapter Pattern

**This is the key integration point.** The adapter wraps any LangChain splitter for use with `SimpleKGPipeline`:

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)

# Wrap LangChain's HTML splitter
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "article"), ("h2", "section"), ("h3", "subsection")]
)

# Adapt for neo4j_graphrag
splitter = LangChainTextSplitterAdapter(html_splitter)

# Use in pipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    text_splitter=splitter,  # LangChain splitter via adapter
    # ...
)
```

### 2.3 Custom TextSplitter for Hierarchical Chunking

The `text_splitter_section.py` example shows how to create a custom splitter:

```python
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

class HierarchicalHTMLSplitter(TextSplitter):
    """Custom splitter for Jama's 3-tier hierarchical chunking."""
    
    def __init__(self, config: HierarchicalChunkingConfig):
        self.config = config
    
    async def run(self, text: str) -> TextChunks:
        chunks = []
        # Level 0: Article summary
        # Level 1: Section chunks
        # Level 2: Sliding window for large sections
        
        # ... implementation ...
        
        return TextChunks(chunks=chunks)
```

### 2.4 Custom DataLoader for HTML Articles

The `data_loader_wikipedia.py` pattern can be adapted for Jama HTML:

```python
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)

class JamaArticleLoader(DataLoader):
    """Load Jama HTML articles for the KG pipeline."""
    
    async def run(self, filepath: Path) -> PdfDocument:
        # Load HTML file
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Extract article metadata
        soup = BeautifulSoup(html_content, "html.parser")
        title = soup.select_one("h1.entry-title").get_text(strip=True)
        
        # Clean and extract main content
        content = self._extract_main_content(soup)
        
        return PdfDocument(
            text=content,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata={
                    "title": title,
                    "url": self._extract_url(soup),
                    "article_id": self._generate_article_id(filepath),
                },
            ),
        )
```

### 2.5 Custom Extraction Prompt with Domain Instructions

The `entity_extraction_prompt.py` pattern is **critical** for accurate entity extraction:

```python
from neo4j_graphrag.generation.prompts import ERExtractionTemplate

# Domain-specific instructions prepended to default template
domain_instructions = """
You are extracting entities from requirements management documentation.

## CRITICAL CLASSIFICATION RULES:

### Industry Entities (label: Industry)
VALID INDUSTRIES - Extract exactly as shown:
- Aerospace & Defense, Automotive, Medical Devices, Pharmaceuticals
- Semiconductor, Financial Services, Telecommunications, Energy
- Industrial Manufacturing, Rail, Government, Software

NOT INDUSTRIES (classify as Concept instead):
- AI, machine learning, cloud computing, IoT, software development
- automation, digital transformation, embedded systems

### Entity Normalization Rules:
- "aerospace and defense" → ONE Industry entity "Aerospace & Defense"
- "medical device" and "medical devices" → same entity
- Preserve acronyms: "DO-178C" not "do-178c"

### Few-Shot Examples:

INPUT: "DO-178C certification is required for aerospace software development."
ENTITIES: [
    {"label": "Standard", "name": "DO-178C"},
    {"label": "Industry", "name": "Aerospace & Defense"}
]
RELATIONSHIPS: [
    {"source": "DO-178C", "type": "APPLIES_TO", "target": "Aerospace & Defense"}
]

INPUT: "AI and machine learning are transforming the automotive industry."
ENTITIES: [
    {"label": "Concept", "name": "artificial intelligence"},
    {"label": "Concept", "name": "machine learning"},
    {"label": "Industry", "name": "Automotive"}
]
RELATIONSHIPS: [
    {"source": "artificial intelligence", "type": "USED_BY", "target": "Automotive"}
]

"""

prompt_template = ERExtractionTemplate(
    template=domain_instructions + ERExtractionTemplate.DEFAULT_TEMPLATE
)
```

### 2.6 LexicalGraphConfig for Custom Labels

Map neo4j_graphrag's default labels to our schema:

```python
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig

config = LexicalGraphConfig(
    chunk_node_label="Chunk",              # Keep as Chunk
    document_node_label="Article",          # Map Document → Article
    chunk_to_document_relationship_type="HAS_CHUNK",  # Reverse direction name
    next_chunk_relationship_type="NEXT",    # Sequential linking
    node_to_chunk_relationship_type="MENTIONS",  # Entity→Chunk provenance
    chunk_embedding_property="embedding",
)
```

---

## 3. Recommendations for Jama ETL Enhancement

### 3.1 Architecture Enhancement: Hybrid Chunking Pipeline

**Replace the current custom HierarchicalChunker with a hybrid approach:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED CHUNKING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: HTML Structure Extraction (LangChain)                             │
│  ─────────────────────────────────────────────                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  HTMLHeaderTextSplitter                                             │   │
│  │  - Split on h1, h2, h3 headers                                      │   │
│  │  - Preserve header hierarchy in metadata                            │   │
│  │  - Output: Document objects with section metadata                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STAGE 2: Size-Based Subsplitting (LangChain)                               │
│  ────────────────────────────────────────────                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RecursiveCharacterTextSplitter (via adapter)                       │   │
│  │  - Applied ONLY to sections > 1500 tokens                           │   │
│  │  - 512 token windows with 64 token overlap                          │   │
│  │  - Preserves parent section metadata                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STAGE 3: Hierarchical Metadata Enrichment (Custom)                         │
│  ──────────────────────────────────────────────────                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JamaHierarchicalEnricher                                           │   │
│  │  - Add level property (0, 1, 2)                                     │   │
│  │  - Add chunk_type ("summary", "section", "window")                  │   │
│  │  - Generate hierarchical chunk IDs                                  │   │
│  │  - Calculate parent_id for relationships                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation: Hybrid Chunker

```python
"""Hybrid chunker combining LangChain HTML splitters with custom enrichment."""

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


class JamaHybridHTMLChunker(TextSplitter):
    """Hybrid chunker: LangChain HTML splitting + custom hierarchical enrichment.
    
    Pipeline:
    1. HTMLHeaderTextSplitter extracts sections by header hierarchy
    2. RecursiveCharacterTextSplitter handles oversized sections
    3. Custom enrichment adds level, chunk_type, parent relationships
    """
    
    def __init__(
        self,
        article_id: str,
        section_max_tokens: int = 1500,
        window_size: int = 512,
        window_overlap: int = 64,
    ):
        self.article_id = article_id
        self.section_max_tokens = section_max_tokens
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        # Stage 1: HTML header splitting
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "article_title"),
                ("h2", "section"),
                ("h3", "subsection"),
            ],
            return_each_element=False,
        )
        
        # Stage 2: Recursive splitting for oversized sections
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_size,
            chunk_overlap=window_overlap,
            length_function=self._count_tokens,
        )
    
    async def run(self, text: str) -> TextChunks:
        chunks = []
        
        # Stage 1: Split by HTML headers
        header_splits = self.html_splitter.split_text(text)
        
        # Stage 2 & 3: Process each section
        level_0_id = f"{self.article_id}-L0"
        
        # Create Level 0 summary chunk (first section or synthesized)
        summary_chunk = self._create_summary_chunk(header_splits, level_0_id)
        chunks.append(summary_chunk)
        
        # Process remaining sections as Level 1
        for i, doc in enumerate(header_splits[1:], start=1):
            section_id = f"{self.article_id}-L1-{i}"
            section_tokens = self._count_tokens(doc.page_content)
            
            # Create Level 1 section chunk
            section_chunk = TextChunk(
                text=doc.page_content,
                index=len(chunks),
                metadata={
                    "id": section_id,
                    "level": 1,
                    "chunk_type": "section",
                    "heading": doc.metadata.get("section", ""),
                    "parent_id": level_0_id,
                    **doc.metadata,  # Include header hierarchy
                },
            )
            chunks.append(section_chunk)
            
            # Stage 2: Apply recursive splitting if section too large
            if section_tokens > self.section_max_tokens:
                window_splits = self.recursive_splitter.split_text(doc.page_content)
                
                for j, window_text in enumerate(window_splits):
                    window_chunk = TextChunk(
                        text=window_text,
                        index=len(chunks),
                        metadata={
                            "id": f"{section_id}-L2-{j}",
                            "level": 2,
                            "chunk_type": "window",
                            "parent_id": section_id,
                            **doc.metadata,
                        },
                    )
                    chunks.append(window_chunk)
        
        return TextChunks(chunks=chunks)
    
    def _create_summary_chunk(self, splits, chunk_id: str) -> TextChunk:
        """Create Level 0 summary from first section or LLM summary."""
        first_section = splits[0] if splits else None
        summary_text = first_section.page_content[:1000] if first_section else ""
        
        return TextChunk(
            text=summary_text,
            index=0,
            metadata={
                "id": chunk_id,
                "level": 0,
                "chunk_type": "summary",
                "heading": first_section.metadata.get("article_title", "") if first_section else "",
            },
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
```

### 3.3 Custom HTML DataLoader

```python
"""Custom DataLoader for Jama HTML articles."""

from pathlib import Path
from bs4 import BeautifulSoup
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)

class JamaHTMLLoader(DataLoader):
    """Load and preprocess Jama HTML articles.
    
    Handles:
    - Main content extraction (strips nav, footer, sidebar)
    - Definition block preservation
    - Resource extraction metadata
    - Article metadata extraction
    """
    
    # Selectors for main content
    CONTENT_SELECTORS = [
        "article .entry-content",
        ".post-content",
        "main article",
    ]
    
    # Elements to remove
    REMOVE_SELECTORS = [
        "nav", "footer", ".sidebar", ".related-articles",
        ".social-share", ".author-bio", "script", "style",
    ]
    
    async def run(self, filepath: Path) -> PdfDocument:
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract metadata
        metadata = self._extract_metadata(soup, filepath)
        
        # Clean and extract main content
        content_html = self._extract_main_content(soup)
        
        return PdfDocument(
            text=content_html,  # Return HTML for HTMLHeaderTextSplitter
            document_info=DocumentInfo(
                path=str(filepath),
                metadata=metadata,
            ),
        )
    
    def _extract_metadata(self, soup: BeautifulSoup, filepath: Path) -> dict:
        """Extract article metadata from HTML."""
        title_elem = soup.select_one("h1.entry-title, h1")
        url_elem = soup.select_one('link[rel="canonical"]')
        
        return {
            "title": title_elem.get_text(strip=True) if title_elem else "",
            "url": url_elem.get("href", "") if url_elem else "",
            "article_id": self._generate_article_id(filepath),
            "source_file": str(filepath),
        }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean main content HTML."""
        # Remove unwanted elements
        for selector in self.REMOVE_SELECTORS:
            for elem in soup.select(selector):
                elem.decompose()
        
        # Find main content
        content = None
        for selector in self.CONTENT_SELECTORS:
            content = soup.select_one(selector)
            if content:
                break
        
        if not content:
            content = soup.body or soup
        
        return str(content)
    
    def _generate_article_id(self, filepath: Path) -> str:
        """Generate article ID from filepath."""
        # e.g., "chapter-1/what-is-requirements-management.html" → "ch1-art1"
        return filepath.stem.replace("-", "_")[:20]
```

### 3.4 Enhanced Extraction Prompt Template

```python
"""Domain-specific extraction prompt for requirements management."""

JAMA_EXTRACTION_PROMPT = """
You are an expert at extracting entities and relationships from requirements 
management documentation produced by Jama Software.

## DOMAIN CONTEXT
This content covers requirements management for regulated industries including:
aerospace, automotive, medical devices, pharmaceuticals, and defense.

## ENTITY CLASSIFICATION RULES

### Industry (label: Industry)
ONLY extract actual industry verticals/market sectors:
✓ VALID: Aerospace & Defense, Automotive, Medical Devices, Pharmaceuticals,
         Semiconductor, Financial Services, Telecommunications, Energy,
         Industrial Manufacturing, Rail, Government, Consumer Products
✗ INVALID (use Concept instead): AI, machine learning, cloud computing, IoT,
         software development, automation, digital transformation, embedded systems

IMPORTANT: "aerospace and defense" = ONE Industry "Aerospace & Defense", not two.

### Standard (label: Standard)  
Regulatory standards and compliance frameworks:
✓ VALID: ISO 13485, DO-178C, IEC 62304, FDA 21 CFR Part 820, ISO 26262,
         ASPICE, MIL-STD-498, IEC 61508, EN 50128

### Concept (label: Concept)
Technical concepts and terminology:
✓ VALID: requirements traceability, verification, validation, baseline management,
         impact analysis, change control, live traceability, bidirectional traceability

### Challenge (label: Challenge)
Problems and difficulties teams face:
✓ VALID: scope creep, requirement ambiguity, poor traceability, compliance gaps,
         change impact uncertainty, audit preparation burden

### Tool (label: Tool)
Software tools and platforms:
✓ VALID: Jama Connect, DOORS, Polarion, Jira, Confluence, ReqIF

### Methodology (label: Methodology)
Development frameworks and approaches:
✓ VALID: Agile, V-Model, MBSE, Waterfall, SAFe, DevOps

## ENTITY NORMALIZATION
- Normalize to lowercase for matching, preserve original case for display
- "medical device" and "medical devices" → same entity
- Preserve acronyms exactly: "DO-178C" not "do 178c"
- "V&V" = "verification and validation" (keep as V&V)

## FEW-SHOT EXAMPLES

### Example 1: Standard applies to Industry
INPUT: "DO-178C certification is critical for aerospace software development."
OUTPUT:
{
  "entities": [
    {"label": "Standard", "name": "DO-178C", "properties": {"organization": "RTCA"}},
    {"label": "Industry", "name": "Aerospace & Defense"}
  ],
  "relationships": [
    {"source": "DO-178C", "target": "Aerospace & Defense", "type": "APPLIES_TO"}
  ]
}

### Example 2: Technology concepts (NOT industries)
INPUT: "AI and machine learning are transforming requirements analysis in automotive."
OUTPUT:
{
  "entities": [
    {"label": "Concept", "name": "artificial intelligence"},
    {"label": "Concept", "name": "machine learning"},
    {"label": "Concept", "name": "requirements analysis"},
    {"label": "Industry", "name": "Automotive"}
  ],
  "relationships": [
    {"source": "artificial intelligence", "target": "requirements analysis", "type": "RELATED_TO"},
    {"source": "machine learning", "target": "Automotive", "type": "USED_BY"}
  ]
}

### Example 3: Practice addresses Challenge
INPUT: "Live traceability eliminates the burden of manual trace matrix updates."
OUTPUT:
{
  "entities": [
    {"label": "Bestpractice", "name": "live traceability"},
    {"label": "Challenge", "name": "manual trace matrix updates"},
    {"label": "Artifact", "name": "trace matrix"}
  ],
  "relationships": [
    {"source": "live traceability", "target": "manual trace matrix updates", "type": "ADDRESSES"},
    {"source": "live traceability", "target": "trace matrix", "type": "PRODUCES"}
  ]
}

### Example 4: Compound industry (ONE entity)
INPUT: "The aerospace and defense sector requires DO-178C and DO-254 compliance."
OUTPUT:
{
  "entities": [
    {"label": "Industry", "name": "Aerospace & Defense"},
    {"label": "Standard", "name": "DO-178C"},
    {"label": "Standard", "name": "DO-254"}
  ],
  "relationships": [
    {"source": "DO-178C", "target": "Aerospace & Defense", "type": "APPLIES_TO"},
    {"source": "DO-254", "target": "Aerospace & Defense", "type": "APPLIES_TO"}
  ]
}

## OUTPUT FORMAT
Return valid JSON with "entities" and "relationships" arrays.
Each entity must have: label, name, and optional properties dict.
Each relationship must have: source, target, type, and optional properties.

"""
```

### 3.5 Dependency Updates

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    
    # LangChain text splitters (HTML-aware)
    "langchain-text-splitters>=0.2.0",
    
    # Required for HTMLSectionSplitter XSLT transformations
    "lxml>=5.0",
]
```

---

## 4. Impact Assessment

### 4.1 Benefits of Proposed Enhancements

| Enhancement | Benefit | Effort |
|-------------|---------|--------|
| LangChain HTMLHeaderTextSplitter | Automatic header hierarchy tracking | LOW |
| LangChainTextSplitterAdapter | Seamless neo4j_graphrag integration | LOW |
| Custom ERExtractionTemplate | 50%+ improvement in entity classification accuracy | MEDIUM |
| JamaHTMLLoader | Clean content extraction, metadata preservation | MEDIUM |
| Hybrid chunking pipeline | Best of both: structure-aware + size-controlled | MEDIUM |

### 4.2 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LangChain version compatibility | Pin to specific version in pyproject.toml |
| HTML structure variations | Fallback to RecursiveCharacterTextSplitter if no headers |
| Extraction prompt too long | Test token count, trim examples if needed |
| Performance overhead | Benchmark LangChain vs custom splitting |

### 4.3 Recommendation Priority

1. **HIGH PRIORITY (Implement First)**
   - Custom `ERExtractionTemplate` with domain instructions and few-shot examples
   - `LangChainTextSplitterAdapter` integration
   - `HTMLHeaderTextSplitter` for structure-aware chunking

2. **MEDIUM PRIORITY (Implement Second)**
   - `JamaHTMLLoader` custom data loader
   - Hybrid chunking pipeline with hierarchical enrichment
   - `LexicalGraphConfig` customization

3. **LOWER PRIORITY (Consider Later)**
   - `HTMLSemanticPreservingSplitter` for table/list preservation
   - Custom XSLT transformations for Avia theme elements

---

## 5. Conclusion

The analysis strongly supports enhancing the Jama ETL pipeline with LangChain's HTML-aware text splitters and the patterns demonstrated in `genai-graphrag-python`. The key recommendations are:

1. **Use `HTMLHeaderTextSplitter`** instead of custom section parsing—it automatically tracks header hierarchy and produces cleaner chunks with metadata.

2. **Integrate via `LangChainTextSplitterAdapter`**—this is the officially supported integration pattern and requires minimal code changes.

3. **Implement domain-specific `ERExtractionTemplate`**—the few-shot examples and classification rules will significantly improve entity extraction accuracy, especially for Industry vs Concept disambiguation.

4. **Create `JamaHTMLLoader`**—centralizes HTML preprocessing and metadata extraction, following the pattern from `data_loader_wikipedia.py`.

5. **Maintain hierarchical structure via metadata enrichment**—rather than building hierarchy into the chunker itself, use LangChain for splitting and a separate enrichment step to add level/parent_id properties.

These enhancements leverage battle-tested LangChain components while maintaining the flexibility needed for Jama's specific content structure and domain requirements.
