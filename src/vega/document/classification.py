"""Rule-based document classification utilities for Vega 2.0."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from .base import (
    BaseDocumentProcessor,
    ConfigurableComponent,
    ConfigurationError,
    DocumentIntelligenceError,
    ProcessingContext,
    ProcessingResult,
    ValidationError,
)

logger = logging.getLogger(__name__)

_WORD_REGEX = re.compile(r"[A-Za-z0-9']+")


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    return " ".join(text.lower().split())


def _derive_keywords_from_name(name: str) -> List[str]:
    """Derive simple keyword list from a category/topic name."""
    tokens = re.split(r"[\s_\-/]+", name.lower())
    keywords = [token for token in tokens if token]

    # Extend with common synonyms for well known domains
    synonyms = {
        "ai": ["artificial intelligence", "machine learning", "neural"],
        "ml": ["machine learning", "deep learning"],
        "legal": ["contract", "agreement", "compliance"],
        "tech": ["technical", "technology"],
        "business": ["enterprise", "corporate", "operations"],
    }

    for token in list(keywords):
        keywords.extend(synonyms.get(token, []))

    # Remove duplicates while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for keyword in keywords:
        key = keyword.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(keyword)
    return deduped or [name.lower()]


def _keyword_score(text: str, keywords: Sequence[str]) -> Tuple[float, List[str]]:
    """Return a score between 0 and 1 together with matched keywords."""
    if not keywords:
        return 0.0, []

    normalized = _normalize_text(text)
    tokens = set(_WORD_REGEX.findall(normalized))
    matches: List[str] = []

    for keyword in keywords:
        key = keyword.lower().strip()
        if not key:
            continue
        if " " in key or "-" in key:
            if key in normalized:
                matches.append(keyword)
        else:
            if key in tokens:
                matches.append(keyword)

    if not matches:
        return 0.0, []

    match_ratio = len(matches) / len(keywords)
    unique_matches = len(set(match.lower() for match in matches))
    diversity_bonus = min(0.25, unique_matches * 0.08)
    density_bonus = min(0.3, len(matches) * 0.05)
    multi_word_bonus = 0.05 * sum(1 for match in matches if " " in match)
    score = min(1.0, match_ratio + diversity_bonus + density_bonus + multi_word_bonus)
    return score, matches


def _resolve_text_payload(
    input_data: Union[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Extract text and options from supported input types."""
    if isinstance(input_data, str):
        text = input_data
        payload: Dict[str, Any] = {}
    elif isinstance(input_data, dict):
        payload = dict(input_data)
        text = payload.get("text")
    else:
        raise ValidationError(
            "Input must be a string or a dictionary containing 'text'."
        )

    if not isinstance(text, str):
        raise ValidationError(
            "Invalid input type: input text must be provided as a string."
        )

    if not text.strip():
        raise ValidationError("Input text cannot be empty.")

    return text, payload


class ClassificationError(DocumentIntelligenceError):
    """Domain specific classification error."""


@dataclass
class ClassificationCategory:
    """Definition of a document classification category."""

    name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    confidence_threshold: Optional[float] = None

    def matches_keywords(self, text: str) -> bool:
        normalized = _normalize_text(text)
        for keyword in self.keywords:
            if keyword.lower() in normalized:
                return True
        return False

    def calculate_score(self, text: str) -> float:
        score, _ = _keyword_score(text, self.keywords)
        return score


@dataclass
class HierarchicalCategory:
    """Simple hierarchical category node."""

    name: str
    level: int
    parent: Optional["HierarchicalCategory"] = None
    subcategories: List["HierarchicalCategory"] = field(default_factory=list)

    def add_subcategory(self, category: "HierarchicalCategory") -> None:
        category.parent = self
        self.subcategories.append(category)

    def get_full_path(self) -> List[str]:
        path: List[str] = []
        node: Optional[HierarchicalCategory] = self
        while node is not None:
            path.append(node.name)
            node = node.parent
        return list(reversed(path))

    def find_by_path(self, path: Sequence[str]) -> Optional["HierarchicalCategory"]:
        if not path:
            return self

        normalized_path = [segment.lower() for segment in path if segment]
        if not normalized_path:
            return self

        node: Optional[HierarchicalCategory] = self
        start_index = 0

        if node.name.lower() == normalized_path[0]:
            if len(normalized_path) == 1:
                return node
            start_index = 1

        for segment in normalized_path[start_index:]:
            if node is None:
                return None
            match = next(
                (
                    child
                    for child in node.subcategories
                    if child.name.lower() == segment
                ),
                None,
            )
            if match is None:
                return None
            node = match

        return node


def _build_default_categories() -> List[ClassificationCategory]:
    """Construct default categories covering common enterprise domains."""
    return [
        ClassificationCategory(
            name="legal",
            description="Contracts, agreements, and legal policy documents",
            keywords=[
                "legal",
                "contract",
                "agreement",
                "liability",
                "compliance",
                "law",
                "clause",
                "terms",
            ],
            confidence_threshold=0.6,
        ),
        ClassificationCategory(
            name="technical",
            description="Technical documentation, APIs, and engineering notes",
            keywords=[
                "api",
                "technical",
                "developer",
                "code",
                "software",
                "documentation",
                "endpoint",
                "integration",
            ],
            confidence_threshold=0.55,
        ),
        ClassificationCategory(
            name="business",
            description="Business strategy, operations, and corporate documents",
            keywords=[
                "business",
                "enterprise",
                "strategy",
                "operations",
                "market",
                "management",
                "stakeholder",
            ],
            confidence_threshold=0.55,
        ),
        ClassificationCategory(
            name="financial",
            description="Invoices, budgets, and financial reports",
            keywords=[
                "invoice",
                "budget",
                "financial",
                "revenue",
                "expense",
                "payment",
                "balance",
            ],
            confidence_threshold=0.55,
        ),
        ClassificationCategory(
            name="research",
            description="Research papers, studies, and academic material",
            keywords=[
                "research",
                "study",
                "paper",
                "experiment",
                "methodology",
                "results",
                "conclusion",
            ],
            confidence_threshold=0.55,
        ),
        ClassificationCategory(
            name="workflow",
            description="Process documentation and procedural instructions",
            keywords=[
                "workflow",
                "procedure",
                "step",
                "process",
                "instructions",
                "review",
                "quality",
            ],
            confidence_threshold=0.5,
        ),
    ]


@dataclass
class ClassificationConfig(ConfigurableComponent):
    """Runtime configuration for document classification."""

    enable_topic_classification: bool = True
    enable_content_classification: bool = True
    enable_hierarchical: bool = True
    enable_intent_classification: bool = True
    min_confidence: float = 0.7
    max_categories: int = 5
    default_categories: List[ClassificationCategory] = field(
        default_factory=_build_default_categories
    )
    fallback_category: str = "general"
    allow_multi_label: bool = True

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if not (0.0 <= self.min_confidence <= 1.0):
            errors.append("min_confidence must be between 0 and 1")
        if self.max_categories <= 0:
            errors.append("max_categories must be greater than 0")
        if not self.default_categories:
            errors.append("default_categories cannot be empty")
        return errors


class ClassificationProcessingResult(ProcessingResult):
    """Processing result with convenience accessors used in tests."""

    @property
    def error(self) -> str:
        return self.errors[0] if self.errors else ""


@dataclass
class ClassificationResult:
    """Aggregate result returned by DocumentClassificationAI."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    context: Optional[ProcessingContext] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None
    error: str = ""

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(message)
            if not self.error:
                self.error = message
        self.success = False

    def add_warning(self, message: str) -> None:
        if message:
            self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


class _BaseKeywordClassifier(BaseDocumentProcessor[ClassificationConfig]):
    """Shared helpers for keyword based classifiers."""

    async def process(
        self, input_data: Any, context: Optional[ProcessingContext] = None
    ) -> ClassificationProcessingResult:
        base_result = await super().process(input_data, context)
        return ClassificationProcessingResult(
            success=base_result.success,
            context=base_result.context,
            data=base_result.data,
            errors=list(base_result.errors),
            warnings=list(base_result.warnings),
            processing_time_ms=base_result.processing_time_ms,
        )

    def _validate_input(self, input_data: Any) -> None:
        _resolve_text_payload(input_data)


class ContentClassifier(_BaseKeywordClassifier):
    """Classify documents into coarse content categories."""

    async def _async_initialize(self) -> None:
        # No heavy initialization required for keyword heuristics
        return None

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        text, options = _resolve_text_payload(input_data)
        normalized = _normalize_text(text)
        categories = options.get("categories", self.config.default_categories)

        multi_label = bool(options.get("multi_label")) or (
            self.config.allow_multi_label and options.get("multi_category")
        )
        max_categories = int(options.get("max_categories", self.config.max_categories))

        scored_categories: List[Dict[str, Any]] = []
        top_score = 0.0
        top_category = self.config.fallback_category

        for category in categories:
            score, matches = _keyword_score(normalized, category.keywords)
            threshold = (
                category.confidence_threshold
                if category.confidence_threshold is not None
                else self.config.min_confidence
            )

            category_entry = {
                "name": category.name,
                "description": category.description,
                "score": round(score, 3),
                "matched_keywords": matches,
                "threshold": threshold,
            }
            scored_categories.append(category_entry)

            if score > top_score:
                top_score = score
                top_category = category.name

        scored_categories.sort(key=lambda item: item["score"], reverse=True)

        selected_categories: List[Dict[str, Any]] = []
        if multi_label:
            for entry in scored_categories:
                if entry["score"] <= 0:
                    continue
                if entry["score"] >= entry["threshold"] or entry["score"] >= 0.2:
                    selected_categories.append(entry)
                if len(selected_categories) >= max_categories:
                    break

        result_data: Dict[str, Any] = {
            "classification": top_category,
            "confidence": round(top_score, 3),
            "top_categories": scored_categories[: max_categories or 1],
            "text_length": len(text),
        }

        if selected_categories:
            result_data["categories"] = selected_categories

        matched_keywords = (
            scored_categories[0]["matched_keywords"] if scored_categories else []
        )
        if matched_keywords:
            result_data["matched_keywords"] = matched_keywords

        return result_data


class TopicClassifier(_BaseKeywordClassifier):
    """Identify thematic topics discussed in a document."""

    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.topic_definitions: List[Dict[str, Any]] = [
            {
                "name": "Artificial Intelligence",
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "ai",
                    "neural",
                    "automation",
                    "deep learning",
                ],
                "description": "Discussions around AI, ML, and automation.",
            },
            {
                "name": "Business Operations",
                "keywords": [
                    "business",
                    "operations",
                    "workflow",
                    "strategy",
                    "organization",
                    "governance",
                ],
                "description": "Operational processes and strategic planning.",
            },
            {
                "name": "Software Development",
                "keywords": [
                    "software",
                    "development",
                    "code",
                    "api",
                    "deployment",
                    "testing",
                ],
                "description": "Engineering and development related topics.",
            },
            {
                "name": "Legal & Compliance",
                "keywords": [
                    "legal",
                    "contract",
                    "agreement",
                    "compliance",
                    "policy",
                    "regulation",
                ],
                "description": "Legal, contractual, and regulatory matters.",
            },
            {
                "name": "Research & Academic",
                "keywords": [
                    "research",
                    "study",
                    "paper",
                    "methodology",
                    "experiment",
                    "results",
                ],
                "description": "Academic or scientific research content.",
            },
        ]

    async def _async_initialize(self) -> None:
        return None

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        text, options = _resolve_text_payload(input_data)
        max_topics = int(options.get("max_topics", min(5, self.config.max_categories)))
        cluster_topics = bool(options.get("cluster_topics"))

        topic_scores: List[Dict[str, Any]] = []
        for topic in self.topic_definitions:
            score, matches = _keyword_score(text, topic["keywords"])
            if score <= 0:
                continue
            topic_scores.append(
                {
                    "name": topic["name"],
                    "relevance": round(score, 3),
                    "keywords": topic["keywords"],
                    "matched_keywords": matches,
                    "description": topic["description"],
                }
            )

        topic_scores.sort(key=lambda entry: entry["relevance"], reverse=True)
        primary_topic = topic_scores[0] if topic_scores else None

        result: Dict[str, Any] = {
            "topics": topic_scores[:max_topics],
            "confidence": primary_topic["relevance"] if primary_topic else 0.0,
        }

        if primary_topic:
            result["primary_topic"] = primary_topic["name"]

        if cluster_topics and topic_scores:
            high_relevance = [
                topic for topic in topic_scores if topic["relevance"] >= 0.5
            ]
            medium_relevance = [
                topic for topic in topic_scores if 0.2 <= topic["relevance"] < 0.5
            ]
            clusters = []
            if high_relevance:
                clusters.append(
                    {
                        "name": "high_relevance",
                        "topics": [t["name"] for t in high_relevance],
                    }
                )
            if medium_relevance:
                clusters.append(
                    {
                        "name": "supporting",
                        "topics": [t["name"] for t in medium_relevance],
                    }
                )
            if clusters:
                result["clusters"] = clusters

        return result


class HierarchicalClassifier(_BaseKeywordClassifier):
    """Produce hierarchical labels for a document."""

    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.hierarchy: Dict[str, Any] = self._build_default_hierarchy()

    async def _async_initialize(self) -> None:
        return None

    def _build_default_hierarchy(self) -> Dict[str, Any]:
        return {
            "legal_documents": {
                "keywords": [
                    "legal",
                    "contract",
                    "agreement",
                    "liability",
                    "policy",
                    "compliance",
                    "document",
                ],
                "children": {
                    "contracts": {
                        "keywords": [
                            "contract",
                            "agreement",
                            "service",
                            "terms",
                        ],
                        "children": {},
                    },
                    "compliance": {
                        "keywords": ["compliance", "regulation", "policy"],
                        "children": {},
                    },
                },
            },
            "business_operations": {
                "keywords": ["business", "enterprise", "operations", "strategy"],
                "children": {
                    "finance": {
                        "keywords": ["invoice", "budget", "financial", "payment"],
                        "children": {},
                    },
                    "governance": {
                        "keywords": ["governance", "oversight", "policy", "compliance"],
                        "children": {},
                    },
                },
            },
            "technology_documents": {
                "keywords": ["technology", "technical", "software", "api", "system"],
                "children": {
                    "software": {
                        "keywords": ["software", "code", "development", "deployment"],
                        "children": {
                            "documentation": {
                                "keywords": ["documentation", "api", "guide", "manual"],
                                "children": {},
                            }
                        },
                    },
                    "operations": {
                        "keywords": ["workflow", "process", "automation", "devops"],
                        "children": {},
                    },
                },
            },
            "research_publications": {
                "keywords": ["research", "study", "paper", "methodology"],
                "children": {
                    "academic": {
                        "keywords": [
                            "abstract",
                            "introduction",
                            "results",
                            "conclusion",
                        ],
                        "children": {},
                    }
                },
            },
        }

    def _normalize_hierarchy(
        self, hierarchy: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, payload in hierarchy.items():
            if isinstance(payload, dict) and "keywords" in payload:
                children = payload.get("children", {})
                normalized[name] = {
                    "keywords": payload.get(
                        "keywords", _derive_keywords_from_name(name)
                    ),
                    "children": (
                        self._normalize_hierarchy(children)
                        if isinstance(children, dict)
                        else {}
                    ),
                }
            elif isinstance(payload, dict):
                normalized[name] = {
                    "keywords": _derive_keywords_from_name(name),
                    "children": self._normalize_hierarchy(payload),
                }
            elif isinstance(payload, list):
                child_dict = {item: {} for item in payload}
                normalized[name] = {
                    "keywords": _derive_keywords_from_name(name),
                    "children": self._normalize_hierarchy(child_dict),
                }
            else:
                normalized[name] = {
                    "keywords": _derive_keywords_from_name(name),
                    "children": {},
                }
        return normalized

    def _resolve_hierarchy_levels(
        self, text: str
    ) -> Tuple[Dict[str, str], float, List[str]]:
        normalized_hierarchy = self._normalize_hierarchy(self.hierarchy)
        normalized_text = _normalize_text(text)
        levels: Dict[str, str] = {}
        matched_keywords: List[str] = []
        scores: List[float] = []

        current_level = normalized_hierarchy
        level_number = 1

        while current_level:
            best_name = None
            best_score = -1.0
            best_matches: List[str] = []

            for name, node in current_level.items():
                score, matches = _keyword_score(
                    normalized_text, node.get("keywords", [])
                )
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_matches = matches

            if best_name is None:
                break

            levels[f"level_{level_number}"] = best_name
            scores.append(max(best_score, 0.0))
            matched_keywords.extend(best_matches)

            current_level = current_level[best_name].get("children", {})
            level_number += 1

        confidence = sum(scores) / len(scores) if scores else 0.0
        return levels, confidence, matched_keywords

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        text, _ = _resolve_text_payload(input_data)
        hierarchy, confidence, matches = self._resolve_hierarchy_levels(text)

        return {
            "hierarchy": hierarchy,
            "confidence": round(confidence, 3),
            "matched_keywords": matches,
            "path": [hierarchy[key] for key in sorted(hierarchy.keys())],
        }


class IntentClassifier(_BaseKeywordClassifier):
    """Detect the author intent of a document."""

    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.intent_definitions: List[Dict[str, Any]] = [
            {
                "name": "request",
                "keywords": [
                    "please",
                    "can you",
                    "could you",
                    "request",
                    "need",
                    "ask",
                    "help",
                    "require",
                ],
                "description": "User is requesting assistance or action.",
            },
            {
                "name": "informational",
                "keywords": [
                    "introduction",
                    "overview",
                    "explains",
                    "describes",
                    "provides",
                    "details",
                    "report",
                ],
                "description": "Document is providing information or context.",
            },
            {
                "name": "action",
                "keywords": [
                    "step",
                    "follow",
                    "complete",
                    "workflow",
                    "process",
                    "review",
                    "execute",
                ],
                "description": "Document contains instructions or actions to perform.",
            },
            {
                "name": "decision",
                "keywords": [
                    "approve",
                    "decision",
                    "select",
                    "choose",
                    "determine",
                ],
                "description": "Document is guiding a decision or approval.",
            },
        ]

    async def _async_initialize(self) -> None:
        return None

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        text, options = _resolve_text_payload(input_data)
        detect_multiple = bool(options.get("detect_multiple"))

        scored_intents: List[Dict[str, Any]] = []
        for intent in self.intent_definitions:
            score, matches = _keyword_score(text, intent["keywords"])
            if score <= 0.0:
                continue
            scored_intents.append(
                {
                    "name": intent["name"],
                    "score": round(score, 3),
                    "matched_keywords": matches,
                    "description": intent["description"],
                }
            )

        scored_intents.sort(key=lambda intent: intent["score"], reverse=True)
        primary_intent = (
            scored_intents[0]
            if scored_intents
            else {
                "name": "informational",
                "score": 0.0,
                "matched_keywords": [],
                "description": "Default informational intent",
            }
        )

        result: Dict[str, Any] = {
            "intent": primary_intent["name"],
            "confidence": primary_intent["score"],
            "matched_keywords": primary_intent["matched_keywords"],
        }

        if detect_multiple and scored_intents:
            threshold = max(0.2, primary_intent["score"] * 0.5)
            result["intents"] = [
                intent for intent in scored_intents if intent["score"] >= threshold
            ]

        return result


class DocumentClassificationAI:
    """High level orchestration for document classification."""

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.content_classifier: Optional[ContentClassifier] = None
        self.topic_classifier: Optional[TopicClassifier] = None
        self.hierarchical_classifier: Optional[HierarchicalClassifier] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        validation_errors = self.config.validate_config()
        if validation_errors:
            raise ConfigurationError(
                f"Invalid classification configuration: {', '.join(validation_errors)}"
            )

        self.content_classifier = ContentClassifier(self.config)
        self.topic_classifier = TopicClassifier(self.config)
        self.hierarchical_classifier = HierarchicalClassifier(self.config)
        self.intent_classifier = IntentClassifier(self.config)

        init_tasks = []
        if self.content_classifier:
            init_tasks.append(self.content_classifier.initialize())
        if self.config.enable_topic_classification and self.topic_classifier:
            init_tasks.append(self.topic_classifier.initialize())
        if self.config.enable_hierarchical and self.hierarchical_classifier:
            init_tasks.append(self.hierarchical_classifier.initialize())
        if self.config.enable_intent_classification and self.intent_classifier:
            init_tasks.append(self.intent_classifier.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks)

        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def classify_document(
        self,
        document: Union[str, Dict[str, Any]],
        context: Optional[ProcessingContext] = None,
    ) -> ClassificationResult:
        await self._ensure_initialized()

        context = context or ProcessingContext()
        result = ClassificationResult(success=True, context=context)
        start_time = asyncio.get_event_loop().time()

        try:
            text, _ = _resolve_text_payload(document)
        except ValidationError as exc:
            result.add_error(str(exc))
            result.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return result

        classification_tasks: List[
            Tuple[str, asyncio.Task[ClassificationProcessingResult]]
        ] = []

        if self.config.enable_content_classification and self.content_classifier:
            classification_tasks.append(
                (
                    "content_classification",
                    asyncio.create_task(
                        self.content_classifier.process(document, context)
                    ),
                )
            )

        if self.config.enable_topic_classification and self.topic_classifier:
            classification_tasks.append(
                (
                    "topic_classification",
                    asyncio.create_task(
                        self.topic_classifier.process(document, context)
                    ),
                )
            )

        if self.config.enable_hierarchical and self.hierarchical_classifier:
            classification_tasks.append(
                (
                    "hierarchical_classification",
                    asyncio.create_task(
                        self.hierarchical_classifier.process(text, context)
                    ),
                )
            )

        if self.config.enable_intent_classification and self.intent_classifier:
            classification_tasks.append(
                (
                    "intent_classification",
                    asyncio.create_task(
                        self.intent_classifier.process(document, context)
                    ),
                )
            )

        if not classification_tasks:
            result.add_error("No classification components are enabled.")
            result.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return result

        gathered = await asyncio.gather(
            *(task for _, task in classification_tasks),
            return_exceptions=True,
        )

        successful_components = 0

        for (name, _task), outcome in zip(classification_tasks, gathered):
            if isinstance(outcome, Exception):
                logger.exception("%s failed during classification", name)
                result.add_error(f"{name.replace('_', ' ')} failed: {outcome}")
                continue

            component_result = cast(ClassificationProcessingResult, outcome)
            if component_result.success:
                successful_components += 1
            else:
                if component_result.error:
                    result.add_error(component_result.error)
                else:
                    result.add_error(f"{name.replace('_', ' ')} failed")

            if name == "content_classification":
                result.data["content_classification_details"] = component_result.data
                if isinstance(component_result.data, dict):
                    result.data[name] = component_result.data.get("classification")
                else:  # pragma: no cover - defensive
                    result.data[name] = component_result.data
            else:
                result.data[name] = component_result.data
            if component_result.warnings:
                result.warnings.extend(component_result.warnings)

        if successful_components == 0 and not result.errors:
            result.add_error("All classification components failed to produce results")

        content_details = result.data.get("content_classification_details", {})
        if isinstance(content_details, dict):
            result.data.setdefault("confidence", content_details.get("confidence", 0.0))
            result.data.setdefault(
                "classification", content_details.get("classification")
            )

        result.processing_time_ms = (
            asyncio.get_event_loop().time() - start_time
        ) * 1000
        return result

    async def classify_topics(
        self,
        document: Union[str, Dict[str, Any]],
        context: Optional[ProcessingContext] = None,
    ) -> ClassificationResult:
        await self._ensure_initialized()
        context = context or ProcessingContext()
        result = ClassificationResult(success=True, context=context)

        if not self.config.enable_topic_classification or not self.topic_classifier:
            result.add_error("Topic classification is disabled")
            return result

        try:
            classifier_result = await self.topic_classifier.process(document, context)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Topic classification failed")
            result.add_error(f"Topic classification failed: {exc}")
            return result

        if classifier_result.success:
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)
        else:
            result.add_error(classifier_result.error or "Topic classification failed")
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)

        result.processing_time_ms = classifier_result.processing_time_ms
        return result

    async def classify_hierarchical(
        self,
        document: Union[str, Dict[str, Any]],
        context: Optional[ProcessingContext] = None,
    ) -> ClassificationResult:
        await self._ensure_initialized()
        context = context or ProcessingContext()
        result = ClassificationResult(success=True, context=context)

        if not self.config.enable_hierarchical or not self.hierarchical_classifier:
            result.add_error("Hierarchical classification is disabled")
            return result

        try:
            text, _ = _resolve_text_payload(document)
            classifier_result = await self.hierarchical_classifier.process(
                text, context
            )
        except ValidationError as exc:
            result.add_error(str(exc))
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Hierarchical classification failed")
            result.add_error(f"Hierarchical classification failed: {exc}")
            return result

        if classifier_result.success:
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)
        else:
            result.add_error(
                classifier_result.error or "Hierarchical classification failed"
            )
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)

        result.processing_time_ms = classifier_result.processing_time_ms
        return result

    async def classify_intent(
        self,
        document: Union[str, Dict[str, Any]],
        context: Optional[ProcessingContext] = None,
    ) -> ClassificationResult:
        await self._ensure_initialized()
        context = context or ProcessingContext()
        result = ClassificationResult(success=True, context=context)

        if not self.config.enable_intent_classification or not self.intent_classifier:
            result.add_error("Intent classification is disabled")
            return result

        try:
            classifier_result = await self.intent_classifier.process(document, context)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Intent classification failed")
            result.add_error(f"Intent classification failed: {exc}")
            return result

        if classifier_result.success:
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)
        else:
            result.add_error(classifier_result.error or "Intent classification failed")
            if isinstance(classifier_result.data, dict):
                result.data.update(classifier_result.data)

        result.processing_time_ms = classifier_result.processing_time_ms
        return result

    async def health_check(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        checks: Dict[str, Any] = {}

        async def _component_health(
            name: str, component: Optional[_BaseKeywordClassifier]
        ):
            if component is None:
                return None
            try:
                return await component.health_check()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Health check failed for %s", name)
                return {"status": "error", "detail": str(exc)}

        checks["content_classifier"] = await _component_health(
            "content_classifier", self.content_classifier
        )
        checks["topic_classifier"] = (
            await _component_health("topic_classifier", self.topic_classifier)
            if self.config.enable_topic_classification
            else None
        )
        checks["hierarchical_classifier"] = (
            await _component_health(
                "hierarchical_classifier", self.hierarchical_classifier
            )
            if self.config.enable_hierarchical
            else None
        )
        checks["intent_classifier"] = (
            await _component_health("intent_classifier", self.intent_classifier)
            if self.config.enable_intent_classification
            else None
        )

        overall_status = "healthy"
        for component in checks.values():
            if component and component.get("status") != "healthy":
                overall_status = "degraded"
                break

        return {"overall_status": overall_status, "components": checks}


__all__ = [
    "ClassificationConfig",
    "ClassificationCategory",
    "ClassificationResult",
    "ClassificationProcessingResult",
    "HierarchicalCategory",
    "ContentClassifier",
    "TopicClassifier",
    "HierarchicalClassifier",
    "IntentClassifier",
    "DocumentClassificationAI",
]
