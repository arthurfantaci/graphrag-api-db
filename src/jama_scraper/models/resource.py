"""Resource models for supplementary graph nodes.

This module defines Pydantic models for resources extracted from articles:
- Images: Diagrams, figures, and graphics
- Videos: Embedded YouTube videos
- Webinars: On-demand webinar resources
- Definitions: Glossary term definitions
- External Links: References to external resources

These become separate nodes in the Neo4j knowledge graph with relationships
to their source articles.
"""

from enum import Enum

from pydantic import BaseModel, Field, computed_field


class ResourceType(str, Enum):
    """Type of resource in the knowledge graph."""

    IMAGE = "image"
    VIDEO = "video"
    WEBINAR = "webinar"
    DEFINITION = "definition"
    EXTERNAL_LINK = "external_link"


class BaseResource(BaseModel):
    """Base class for all resource types.

    Provides common fields for identification and provenance tracking.

    Attributes:
        resource_id: Unique identifier for this resource.
        resource_type: Discriminator for resource subtype.
        source_article_id: Article where this resource appears.
        url: Primary URL for the resource.
    """

    resource_id: str = Field(description="Unique resource identifier")
    resource_type: ResourceType = Field(description="Type of resource")
    source_article_id: str = Field(description="Article ID where this resource appears")
    url: str = Field(description="Primary URL for the resource")


class ImageResource(BaseResource):
    """An image or diagram embedded in an article.

    Captures diagrams, figures, and graphics with their contextual
    metadata for knowledge graph construction.

    Attributes:
        alt_text: Accessibility description of the image.
        title: Optional title attribute.
        caption: Figure caption if present.
        context: Surrounding text context (for understanding relevance).
        width: Image width in pixels if specified.
        height: Image height in pixels if specified.
    """

    resource_type: ResourceType = Field(
        default=ResourceType.IMAGE, description="Resource type discriminator"
    )
    alt_text: str = Field(default="", description="Accessibility description")
    title: str | None = Field(default=None, description="Title attribute")
    caption: str | None = Field(default=None, description="Figure caption")
    context: str | None = Field(default=None, description="Surrounding text context")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")

    @computed_field
    @property
    def has_caption(self) -> bool:
        """Whether this image has a caption."""
        return self.caption is not None and len(self.caption) > 0

    @computed_field
    @property
    def is_figure(self) -> bool:
        """Whether this appears to be a labeled figure vs inline image."""
        return self.has_caption or (
            self.alt_text
            and any(
                keyword in self.alt_text.lower()
                for keyword in ["figure", "diagram", "chart", "graph", "flow"]
            )
        )


class VideoResource(BaseResource):
    """An embedded video (typically YouTube) in an article.

    Captures video metadata for knowledge graph construction.

    Attributes:
        video_id: Platform-specific video identifier.
        platform: Video hosting platform (youtube, vimeo, etc.).
        embed_url: URL for embedding the video.
        title: Video title if available.
        duration_seconds: Video duration if known.
        context: Surrounding text context.
    """

    resource_type: ResourceType = Field(
        default=ResourceType.VIDEO, description="Resource type discriminator"
    )
    video_id: str = Field(description="Platform video ID (e.g., YouTube video ID)")
    platform: str = Field(default="youtube", description="Video platform")
    embed_url: str = Field(description="Video embed URL")
    title: str | None = Field(default=None, description="Video title")
    duration_seconds: int | None = Field(
        default=None, description="Duration in seconds"
    )
    context: str | None = Field(default=None, description="Surrounding text context")

    @computed_field
    @property
    def watch_url(self) -> str:
        """Generate the watch URL for the video."""
        if self.platform == "youtube":
            return f"https://www.youtube.com/watch?v={self.video_id}"
        if self.platform == "vimeo":
            return f"https://vimeo.com/{self.video_id}"
        return self.url


class WebinarResource(BaseResource):
    """A webinar resource linked from an article.

    Captures on-demand webinar resources with their promotional
    metadata for knowledge graph construction.

    Attributes:
        title: Webinar title or promotional text.
        description: "In This Webinar" section content if available.
        thumbnail_url: Preview image URL.
        presenter: Webinar presenter if known.
        context: Surrounding heading/promotional context.
    """

    resource_type: ResourceType = Field(
        default=ResourceType.WEBINAR, description="Resource type discriminator"
    )
    title: str = Field(description="Webinar title")
    description: str | None = Field(
        default=None, description="Webinar description or summary"
    )
    thumbnail_url: str | None = Field(default=None, description="Thumbnail image URL")
    presenter: str | None = Field(default=None, description="Webinar presenter name")
    context: str | None = Field(default=None, description="Surrounding context")

    @computed_field
    @property
    def has_description(self) -> bool:
        """Whether this webinar has a full description."""
        return self.description is not None and len(self.description) > 50


class DefinitionResource(BaseResource):
    """A glossary term definition.

    Captures glossary definitions as resources that can be linked
    to concepts extracted from articles.

    Attributes:
        term: The term being defined.
        definition: The definition text.
        acronym: Acronym if applicable.
        related_terms: Related glossary terms.
        related_concept_ids: Extracted Concept entities this defines.
    """

    resource_type: ResourceType = Field(
        default=ResourceType.DEFINITION, description="Resource type discriminator"
    )
    term: str = Field(description="The term being defined")
    definition: str = Field(description="Definition text")
    acronym: str | None = Field(default=None, description="Acronym if applicable")
    related_terms: list[str] = Field(
        default_factory=list, description="Related glossary terms"
    )
    related_concept_ids: list[str] = Field(
        default_factory=list, description="Linked Concept entity IDs"
    )

    @computed_field
    @property
    def definition_length(self) -> int:
        """Length of the definition in words."""
        return len(self.definition.split())


class ExternalLinkResource(BaseResource):
    """An external link reference from an article.

    Captures links to external resources like standards documents,
    tools, or reference materials.

    Attributes:
        link_text: The anchor text of the link.
        domain: The domain of the external URL.
        link_type: Classification of the link type.
        context: Surrounding text context.
    """

    resource_type: ResourceType = Field(
        default=ResourceType.EXTERNAL_LINK, description="Resource type discriminator"
    )
    link_text: str = Field(description="Anchor text of the link")
    domain: str = Field(description="Domain of the external URL")
    link_type: str = Field(
        default="reference",
        description="Link classification (standard, tool, reference, vendor)",
    )
    context: str | None = Field(default=None, description="Surrounding text context")

    @computed_field
    @property
    def is_standard(self) -> bool:
        """Whether this links to a standards document."""
        standard_domains = {
            "iso.org",
            "ieee.org",
            "incose.org",
            "faa.gov",
            "fda.gov",
            "rtca.org",
        }
        return any(std in self.domain.lower() for std in standard_domains)
