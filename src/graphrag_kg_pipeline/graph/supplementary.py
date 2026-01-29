"""Supplementary graph structure builder.

This module creates supplementary nodes and relationships that
complement the entity extraction:
- Chapter nodes with hierarchy
- Resource nodes (Image, Video, Webinar)
- Glossary Definition nodes
- Article-to-article relationships
"""

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import Driver

    from graphrag_kg_pipeline.models import RequirementsManagementGuide

logger = structlog.get_logger(__name__)


class SupplementaryGraphBuilder:
    """Builder for supplementary graph structure.

    Creates additional nodes and relationships that are not
    extracted by the LLM pipeline but are important for
    the knowledge graph structure.

    Example:
        >>> builder = SupplementaryGraphBuilder(driver)
        >>> stats = await builder.build_all(guide)
        >>> print(f"Created {stats['chapters']} chapter nodes")
    """

    def __init__(self, driver: "Driver", database: str = "neo4j") -> None:
        """Initialize the builder.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def build_all(
        self,
        guide: "RequirementsManagementGuide",
    ) -> dict[str, Any]:
        """Build all supplementary graph structure.

        Args:
            guide: The scraped guide.

        Returns:
            Statistics about created nodes/relationships.
        """
        stats = {
            "chapters": 0,
            "images": 0,
            "videos": 0,
            "webinars": 0,
            "definitions": 0,
            "article_relationships": 0,
        }

        # Create chapter structure
        chapter_stats = await create_chapter_structure(
            self.driver, guide, self.database
        )
        stats["chapters"] = chapter_stats.get("chapters_created", 0)

        # Create resource nodes
        resource_stats = await create_resource_nodes(self.driver, guide, self.database)
        stats["images"] = resource_stats.get("images", 0)
        stats["videos"] = resource_stats.get("videos", 0)
        stats["webinars"] = resource_stats.get("webinars", 0)

        # Create glossary structure
        if guide.glossary:
            glossary_stats = await create_glossary_structure(
                self.driver, guide.glossary, self.database
            )
            stats["definitions"] = glossary_stats.get("definitions", 0)

        # Create article relationships
        rel_stats = await create_article_relationships(
            self.driver, guide, self.database
        )
        stats["article_relationships"] = rel_stats.get("relationships", 0)

        logger.info(
            "Supplementary graph complete",
            **stats,
        )

        return stats


async def create_chapter_structure(
    driver: "Driver",
    guide: "RequirementsManagementGuide",
    database: str = "neo4j",
) -> dict[str, int]:
    """Create Chapter nodes and IN_CHAPTER relationships.

    Args:
        driver: Neo4j driver.
        guide: The scraped guide.
        database: Database name.

    Returns:
        Statistics about created nodes.
    """
    stats = {"chapters_created": 0, "relationships_created": 0}

    async with driver.session(database=database) as session:
        for chapter in guide.chapters:
            # Create Chapter node
            create_chapter_query = """
            MERGE (ch:Chapter {chapter_number: $chapter_number})
            SET ch.title = $title,
                ch.overview_url = $overview_url,
                ch.article_count = $article_count
            RETURN ch
            """

            await session.run(
                create_chapter_query,
                chapter_number=chapter.chapter_number,
                title=chapter.title,
                overview_url=chapter.overview_url,
                article_count=len(chapter.articles),
            )
            stats["chapters_created"] += 1

            # Link articles to chapter
            link_articles_query = """
            MATCH (ch:Chapter {chapter_number: $chapter_number})
            MATCH (a:Article {chapter_number: $chapter_number})
            MERGE (a)-[:IN_CHAPTER]->(ch)
            RETURN count(*) AS linked
            """

            result = await session.run(
                link_articles_query,
                chapter_number=chapter.chapter_number,
            )
            record = await result.single()
            if record:
                stats["relationships_created"] += record["linked"]

    logger.info(
        "Created chapter structure",
        chapters=stats["chapters_created"],
        relationships=stats["relationships_created"],
    )

    return stats


async def create_resource_nodes(
    driver: "Driver",
    guide: "RequirementsManagementGuide",
    database: str = "neo4j",
) -> dict[str, int]:
    """Create resource nodes (Image, Video, Webinar) from articles.

    Args:
        driver: Neo4j driver.
        guide: The scraped guide.
        database: Database name.

    Returns:
        Statistics about created nodes.
    """
    stats = {"images": 0, "videos": 0, "webinars": 0}

    async with driver.session(database=database) as session:
        for chapter in guide.chapters:
            for article in chapter.articles:
                # Create Image nodes
                for i, image in enumerate(article.images):
                    image_id = f"{article.article_id}-img{i}"

                    query = """
                    MERGE (img:Image {resource_id: $resource_id})
                    SET img.url = $url,
                        img.alt_text = $alt_text,
                        img.caption = $caption,
                        img.context = $context,
                        img.source_article_id = $article_id
                    WITH img
                    MATCH (a:Article {article_id: $article_id})
                    MERGE (a)-[:HAS_IMAGE]->(img)
                    """

                    await session.run(
                        query,
                        resource_id=image_id,
                        url=image.url,
                        alt_text=image.alt_text,
                        caption=image.caption,
                        context=image.context,
                        article_id=article.article_id,
                    )
                    stats["images"] += 1

                # Create Video nodes
                for i, video in enumerate(article.videos):
                    video_id = f"{article.article_id}-vid{i}"

                    query = """
                    MERGE (vid:Video {resource_id: $resource_id})
                    SET vid.url = $url,
                        vid.video_id = $video_platform_id,
                        vid.platform = $platform,
                        vid.embed_url = $embed_url,
                        vid.title = $title,
                        vid.context = $context,
                        vid.source_article_id = $article_id
                    WITH vid
                    MATCH (a:Article {article_id: $article_id})
                    MERGE (a)-[:HAS_VIDEO]->(vid)
                    """

                    await session.run(
                        query,
                        resource_id=video_id,
                        url=video.url,
                        video_platform_id=video.video_id,
                        platform=video.platform,
                        embed_url=video.embed_url,
                        title=video.title,
                        context=video.context,
                        article_id=article.article_id,
                    )
                    stats["videos"] += 1

                # Create Webinar nodes
                for i, webinar in enumerate(article.webinars):
                    webinar_id = f"{article.article_id}-web{i}"

                    query = """
                    MERGE (web:Webinar {resource_id: $resource_id})
                    SET web.url = $url,
                        web.title = $title,
                        web.description = $description,
                        web.thumbnail_url = $thumbnail_url,
                        web.context = $context,
                        web.source_article_id = $article_id
                    WITH web
                    MATCH (a:Article {article_id: $article_id})
                    MERGE (a)-[:HAS_WEBINAR]->(web)
                    """

                    await session.run(
                        query,
                        resource_id=webinar_id,
                        url=webinar.url,
                        title=webinar.title,
                        description=webinar.description,
                        thumbnail_url=webinar.thumbnail_url,
                        context=webinar.context,
                        article_id=article.article_id,
                    )
                    stats["webinars"] += 1

    logger.info(
        "Created resource nodes",
        images=stats["images"],
        videos=stats["videos"],
        webinars=stats["webinars"],
    )

    return stats


async def create_glossary_structure(
    driver: "Driver",
    glossary: Any,
    database: str = "neo4j",
) -> dict[str, int]:
    """Create Definition nodes from glossary.

    Args:
        driver: Neo4j driver.
        glossary: Glossary object.
        database: Database name.

    Returns:
        Statistics about created nodes.
    """
    stats = {"definitions": 0, "related_links": 0}

    async with driver.session(database=database) as session:
        for term in glossary.terms:
            # Create Definition node
            term_id = term.term.lower().replace(" ", "_")

            query = """
            MERGE (d:Definition {term_id: $term_id})
            SET d.term = $term,
                d.definition = $definition,
                d.acronym = $acronym,
                d.url = $url
            """

            await session.run(
                query,
                term_id=term_id,
                term=term.term,
                definition=term.definition,
                acronym=term.acronym,
                url=glossary.url,
            )
            stats["definitions"] += 1

            # Link to related chapters
            if term.related_chapters:
                for chapter_num in term.related_chapters:
                    link_query = """
                    MATCH (d:Definition {term_id: $term_id})
                    MATCH (ch:Chapter {chapter_number: $chapter_number})
                    MERGE (d)-[:RELEVANT_TO]->(ch)
                    """

                    await session.run(
                        link_query,
                        term_id=term_id,
                        chapter_number=chapter_num,
                    )
                    stats["related_links"] += 1

    logger.info(
        "Created glossary structure",
        definitions=stats["definitions"],
        related_links=stats["related_links"],
    )

    return stats


async def create_article_relationships(
    driver: "Driver",
    guide: "RequirementsManagementGuide",
    database: str = "neo4j",
) -> dict[str, int]:
    """Create relationships between articles based on cross-references.

    Args:
        driver: Neo4j driver.
        guide: The scraped guide.
        database: Database name.

    Returns:
        Statistics about created relationships.
    """
    stats = {"relationships": 0}

    # Build URL to article_id mapping
    url_to_id: dict[str, str] = {}
    for chapter in guide.chapters:
        for article in chapter.articles:
            url_to_id[article.url] = article.article_id

    async with driver.session(database=database) as session:
        for chapter in guide.chapters:
            for article in chapter.articles:
                # Check cross-references
                for ref in article.cross_references:
                    if ref.is_internal and ref.url in url_to_id:
                        target_id = url_to_id[ref.url]

                        if target_id != article.article_id:
                            query = """
                            MATCH (source:Article {article_id: $source_id})
                            MATCH (target:Article {article_id: $target_id})
                            MERGE (source)-[:REFERENCES {text: $text}]->(target)
                            """

                            await session.run(
                                query,
                                source_id=article.article_id,
                                target_id=target_id,
                                text=ref.text,
                            )
                            stats["relationships"] += 1

                # Check related articles
                for related in article.related_articles:
                    if related.url in url_to_id:
                        target_id = url_to_id[related.url]

                        if target_id != article.article_id:
                            query = """
                            MATCH (source:Article {article_id: $source_id})
                            MATCH (target:Article {article_id: $target_id})
                            MERGE (source)-[:RELATED_TO {title: $title}]->(target)
                            """

                            await session.run(
                                query,
                                source_id=article.article_id,
                                target_id=target_id,
                                title=related.title,
                            )
                            stats["relationships"] += 1

    logger.info(
        "Created article relationships",
        relationships=stats["relationships"],
    )

    return stats
