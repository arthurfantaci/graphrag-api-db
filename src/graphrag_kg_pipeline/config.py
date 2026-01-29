"""Configuration and URL mappings for the Jama Requirements Management Guide.

All URLs extracted from the guide's table of contents.
"""

from dataclasses import dataclass, field

BASE_URL = "https://www.jamasoftware.com/requirements-management-guide"


@dataclass
class ArticleConfig:
    """Configuration for a single article."""

    number: int  # 0 for overview
    title: str
    slug: str  # URL path segment


@dataclass
class ChapterConfig:
    """Configuration for a chapter."""

    number: int
    title: str
    slug: str
    articles: list[ArticleConfig] = field(default_factory=list)

    @property
    def overview_url(self) -> str:
        """Return the URL for this chapter's overview page."""
        return f"{BASE_URL}/{self.slug}/"

    def get_article_url(self, article: ArticleConfig) -> str:
        """Return the full URL for an article within this chapter."""
        if article.number == 0:
            return self.overview_url
        return f"{BASE_URL}/{self.slug}/{article.slug}/"


# Complete chapter and article configuration
CHAPTERS: list[ChapterConfig] = [
    ChapterConfig(
        number=1,
        title="Requirements Management",
        slug="requirements-management",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1, "What is Requirements Management?", "what-is-requirements-management"
            ),
            ArticleConfig(
                2,
                "Why do you need Requirements Management?",
                "why-do-you-need-requirements-management",
            ),
            ArticleConfig(
                3,
                "Four Stages of Requirements Management Processes",
                "four-fundamentals-of-requirements-management",
            ),
            ArticleConfig(
                4,
                "Adopting an Agile Approach to Requirements Management",
                "adopting-an-agile-approach-to-requirements-management",
            ),
            ArticleConfig(5, "Status Request Changes", "status-requests-changes"),
            ArticleConfig(
                6,
                "Conquering the 5 Biggest Challenges of Requirements Management",
                "conquering-the-5-biggest-challenges-of-requirements-management",
            ),
            ArticleConfig(
                7,
                "Three Reasons You Need a Requirements Management Solution",
                "three-reasons-you-need-a-requirements-management-solution",
            ),
            ArticleConfig(
                8,
                "Guide to Poor Requirements: Identify Causes, Repercussions, and How to Fix Them",
                "guide-to-poor-requirements-identify-causes-repercussions-and-how-to-fix-them",
            ),
        ],
    ),
    ChapterConfig(
        number=2,
        title="Writing Requirements",
        slug="writing-requirements",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1,
                "Functional requirements examples and templates",
                "functional-requirements-examples-and-templates",
            ),
            ArticleConfig(
                2,
                "Identifying and Measuring Requirements Quality",
                "identifying-and-measuring-the-quality-of-requirements",
            ),
            ArticleConfig(
                3,
                "How to write system requirement specification (SRS) documents",
                "how-to-write-system-requirement-specification-srs-documents",
            ),
            ArticleConfig(
                4,
                "The Fundamentals of Business Requirements",
                "the-fundamentals-of-business-requirements-examples-of-business-requirements-and-the-importance-of-excellence",
            ),
            ArticleConfig(
                5,
                "Adopting the EARS Notation to Improve Requirements Engineering",
                "adopting-the-ears-notation-to-improve-requirements-engineering",
            ),
            ArticleConfig(6, "Jama Connect Advisor™", "jama-connect-advisor"),
            ArticleConfig(
                7,
                "FAQ: EARS Notation and Jama Connect Advisor™",
                "frequently-asked-questions-about-the-ears-notation-and-jama-connect-requirements-advisor",
            ),
            ArticleConfig(
                8,
                "How to Write an Effective Product Requirements Document (PRD)",
                "how-to-write-an-effective-product-requirements-document",
            ),
            ArticleConfig(
                9,
                "Functional vs. Non-Functional Requirements",
                "functional-vs-non-functional-requirements",
            ),
            ArticleConfig(
                10,
                "What Are Nonfunctional Requirements and How Do They Impact Product Development?",
                "how-non-functional-requirements-impact-product-development",
            ),
            ArticleConfig(
                11,
                "Characteristics of Effective Software Requirements and SRS",
                "the-characteristics-of-excellent-requirements",
            ),
            ArticleConfig(
                12,
                "8 Do's and Don'ts for Writing Requirements",
                "8-dos-and-donts-for-writing-requirements",
            ),
        ],
    ),
    ChapterConfig(
        number=3,
        title="Requirements Gathering and Management Processes",
        slug="requirements-gathering-and-management-processes",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(1, "Requirements Engineering", "requirements-engineering"),
            ArticleConfig(2, "Requirements Analysis", "requirements-analysis"),
            ArticleConfig(
                3,
                "A Guide to Requirements Elicitation for Product Teams",
                "a-guide-to-requirements-elicitation-for-product-teams",
            ),
            ArticleConfig(
                4,
                "Requirements Gathering Techniques for Agile Product Teams",
                "11-requirements-gathering-techniques-for-agile-product-teams",
            ),
            ArticleConfig(
                5, "What is Requirements Gathering?", "what-is-requirements-gathering"
            ),
            ArticleConfig(
                6,
                "Defining and Implementing a Requirements Baseline",
                "defining-and-implementing-requirements-baselines",
            ),
            ArticleConfig(
                7,
                "Managing Project Scope — Why It Matters and Best Practices",
                "managing-project-scope-why-it-matters-and-best-practices",
            ),
            ArticleConfig(
                8, "How Long Do Requirements Take?", "how-long-do-requirements-take"
            ),
            ArticleConfig(
                9,
                "How to Reuse Requirements Across Multiple Products",
                "how-to-reuse-requirements-across-multiple-products",
            ),
        ],
    ),
    ChapterConfig(
        number=4,
        title="Requirements Traceability",
        slug="requirements-traceability",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(1, "What is Traceability?", "what-is-traceability"),
            ArticleConfig(
                2,
                "How is Traceability Achieved? A Practical Guide for Engineers",
                "how-is-traceability-achieved-a-practical-guide-for-engineers",
            ),
            ArticleConfig(
                3,
                "Tracing Your Way to Success: The Crucial Role of Traceability",
                "tracing-your-way-to-success-the-crucial-role-of-traceability-in-modern-product-and-systems-development",
            ),
            # Note: This URL has a typo in the original site ("reguirements" instead of "requirements")
            ArticleConfig(
                4,
                "Change Impact Analysis (CIA): A Short Guide",
                "change-impact-analysis-cia-a-short-guide-for-effective-implementation",
            ),
            ArticleConfig(
                5,
                "What is Requirements Traceability and Why Does It Matter?",
                "what-is-traceability-and-why-does-it-matter-for-product-teams",
            ),
            ArticleConfig(
                6,
                "What is Meant by Version Control?",
                "what-is-meant-by-version-control",
            ),
            ArticleConfig(
                7,
                "Key Traceability Challenges and Tips",
                "key-traceability-challenges-and-tips-for-ensuring-accountability-and-efficiency",
            ),
            ArticleConfig(
                8,
                "Unraveling the Digital Thread",
                "unraveling-the-digital-thread-enhancing-connectivity-and-efficiency",
            ),
            ArticleConfig(
                9,
                "The Role of a Data Thread in Product and Software Development",
                "the-role-of-a-data-thread-in-product-and-software-development",
            ),
            ArticleConfig(
                10,
                "How to Create and Use a Requirements Traceability Matrix",
                "how-to-create-and-use-a-requirements-traceability-matrix",
            ),
            ArticleConfig(
                11,
                "Traceability Matrix 101: Why It's Not the Ultimate Solution",
                "traceability-matrix-101-why-its-not-the-ultimate-solution-for-managing-requirements",
            ),
            ArticleConfig(
                12,
                "Live Traceability vs. After-the-Fact Traceability",
                "live-traceability-vs-after-the-fact-traceability",
            ),
            ArticleConfig(
                13,
                "How to Overcome Organizational Barriers to Live Requirements Traceability",
                "how-to-overcome-organizational-barriers-to-live-requirements-traceability",
            ),
            ArticleConfig(
                14,
                "Requirements Traceability, What Are You Missing?",
                "requirements-traceability-what-are-you-missing",
            ),
            ArticleConfig(
                15,
                "Four Best Practices for Requirements Traceability",
                "four-best-practices-for-requirements-traceability",
            ),
            ArticleConfig(
                16,
                "Requirements Traceability: Links in the Chain",
                "links-in-the-chain",
            ),
            ArticleConfig(
                17,
                "What Are the Benefits of End-to-End Traceability?",
                "what-are-the-benefits-of-end-to-end-traceability-in-product-development",
            ),
        ],
    ),
    ChapterConfig(
        number=5,
        title="Requirements Management Tools and Software",
        slug="requirements-management-tools-and-software",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1,
                "Selecting the Right Requirements Management Tools and Software",
                "selecting-the-right-requirements-management-tools-and-software",
            ),
            ArticleConfig(
                2,
                "Why Investing in RM Software Makes Business Sense",
                "why-investing-in-rm-software-makes-good-business-sense",
            ),
            ArticleConfig(
                3,
                "Why Word and Excel Alone is Not Enough",
                "why-word-and-excel-alone-is-not-enough-for-product-software-and-systems-development",
            ),
            ArticleConfig(
                4,
                "Application Lifecycle Management (ALM)",
                "application-lifecycle-management-alm",
            ),
            ArticleConfig(
                5, "Is There Life After DOORS®?", "is-there-life-after-doors"
            ),
            ArticleConfig(
                6,
                "Can You Track Requirements in Jira?",
                "can-you-track-requirements-in-jira",
            ),
            ArticleConfig(
                7,
                "Checklist: Selecting a Requirements Management Tool",
                "checklist-selecting-a-requirements-management-tool",
            ),
        ],
    ),
    ChapterConfig(
        number=6,
        title="Requirements Validation and Verification",
        slug="requirements-validation-and-verification",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=7,
        title="Meeting Regulatory Compliance and Industry Standards",
        slug="meeting-regulatory-compliance-and-industry-standards",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1, "Understanding ISO Standards", "understanding-iso-standards"
            ),
            ArticleConfig(
                2,
                "Understanding ISO/IEC 27001",
                "understanding-iso-iec-27001-a-guide-to-information-security-management",
            ),
            ArticleConfig(
                3,
                "What is DevSecOps? A Guide to Building Secure Software",
                "what-is-devsecops-a-guide-to-building-secure-software",
            ),
            ArticleConfig(4, "Compliance Management", "compliance-management"),
            ArticleConfig(
                5, "What is FMEA? Failure Modes and Effects Analysis", "fmea"
            ),
            ArticleConfig(
                6,
                "TÜV SÜD: Ensuring Safety, Quality, and Sustainability",
                "tuv-sud-ensuring-safety-quality-and-sustainability-worldwide",
            ),
        ],
    ),
    ChapterConfig(
        number=8,
        title="Systems Engineering",
        slug="systems-engineering",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1, "What is Systems Engineering?", "what-is-systems-engineering"
            ),
            ArticleConfig(
                2,
                "How Do Engineers Collaborate?",
                "how-do-engineers-collaborate-a-guide-to-streamlined-teamwork-and-innovation",
            ),
            ArticleConfig(
                3,
                "The Systems Engineering Body of Knowledge (SEBoK)",
                "the-systems-engineering-body-of-knowledge-sebok",
            ),
            ArticleConfig(
                4,
                "What is MBSE? Model-Based Systems Engineering Explained",
                "what-is-mbse-model-based-systems-engineering-explained",
            ),
            ArticleConfig(
                5,
                "Digital Engineering Between Government and Contractors",
                "digital-engineering-between-government-and-contractors",
            ),
            ArticleConfig(
                6,
                "Digital Engineering Tools",
                "digital-engineering-tools-the-key-to-driving-innovation-and-efficiency-in-complex-systems",
            ),
        ],
    ),
    ChapterConfig(
        number=9,
        title="Automotive Development",
        slug="automotive-engineering",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=10,
        title="Medical Device & Life Sciences Development",
        slug="medical-devices",
        articles=[
            ArticleConfig(0, "Overview", ""),
            ArticleConfig(
                1,
                "The Importance of Benefit-Risk Analysis in Medical Device Development",
                "the-importance-of-benefit-risk-analysis-in-medical-device-development",
            ),
            ArticleConfig(
                2,
                "Software as a Medical Device: Revolutionizing Healthcare",
                "software-as-a-medical-device-revolutionizing-healthcare",
            ),
            ArticleConfig(
                3, "What's a Design History File (DHF)?", "design-history-file-dhf"
            ),
            ArticleConfig(
                4,
                "Navigating the Risks of SOUP",
                "navigating-the-risks-of-software-of-unknown-pedigree-soup-in-the-medical-device-and-life-sciences-industry",
            ),
            ArticleConfig(5, "What is ISO 13485?", "iso-13485"),
            ArticleConfig(
                6,
                "ANSI/AAMI SW96:2023 — Medical Device Security",
                "what-you-need-to-know-ansi-aami-sw96-2023-medical-device-security",
            ),
            ArticleConfig(
                7,
                "ISO 13485 vs ISO 9001",
                "iso-13485-vs-iso-9001-understanding-the-differences-and-synergies",
            ),
            ArticleConfig(
                8,
                "FMEDA for Medical Devices",
                "failure-modes-effects-and-diagnositc-analysis-fmeda-for-medical-devices-what-you-need-to-know",
            ),
            ArticleConfig(
                9,
                "Internet of Medical Things (IoMT)",
                "embracing-the-future-of-healthcare-exploring-the-internet-of-medical-things-iomt",
            ),
        ],
    ),
    ChapterConfig(
        number=11,
        title="Aerospace & Defense Development",
        slug="aerospace-and-defense",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=12,
        title="Architecture, Engineering, and Construction (AEC)",
        slug="architecture-engineering-and-construction-aec-development",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=13,
        title="Industrial Manufacturing & Machinery, Automation & Robotics, Consumer Electronics, and Energy",
        slug="industrial-manufacturing-development",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=14,
        title="Semiconductor Development",
        slug="semiconductor",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
    ChapterConfig(
        number=15,
        title="AI in Product Development",
        slug="artificial-intelligence-in-product-development",
        articles=[
            ArticleConfig(0, "Overview", ""),
            # Note: Need to discover sub-articles by scraping overview page
        ],
    ),
]

GLOSSARY_URL = f"{BASE_URL}/rm-glossary/"

# URL variations to handle (some pages have typos in the original site)
URL_TYPO_FIXES = {
    "reguirements-traceability": "requirements-traceability",  # Chapter 4, article 4
}

# Rate limiting configuration
RATE_LIMIT_DELAY_SECONDS = 1.0  # Delay between requests to be respectful
MAX_CONCURRENT_REQUESTS = 3  # Max parallel requests
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
