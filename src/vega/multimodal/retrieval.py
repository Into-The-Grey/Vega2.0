"""
Unified Multi-Modal Retrieval System
====================================

Advanced retrieval system that provides unified access to multi-modal content
through semantic search, filtering, and ranking capabilities.

Features:
- Cross-modal semantic search
- Relevance ranking and re-ranking
- Query expansion and refinement
- Result aggregation and fusion
- Context-aware retrieval
- Personalization and user preferences
"""

import asyncio
import logging
import numpy as np
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from collections import defaultdict, Counter
import heapq

# Local imports
from .cross_modal_search import (
    CrossModalSearchEngine,
    SearchQuery,
    SearchResult,
    ModalityType,
    SimilarityMetric,
    ContentItem,
)
from .embeddings import MultiModalEmbeddings, EmbeddingConfig, EmbeddingModel

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Different retrieval strategies"""

    SIMILARITY_ONLY = "similarity_only"  # Pure similarity search
    HYBRID = "hybrid"  # Combination of similarity + filters
    CONTEXTUAL = "contextual"  # Context-aware retrieval
    PERSONALIZED = "personalized"  # User preference-based
    MULTI_STAGE = "multi_stage"  # Multi-stage retrieval pipeline


class RankingMethod(Enum):
    """Ranking methods for results"""

    SIMILARITY_SCORE = "similarity_score"
    RELEVANCE_SCORE = "relevance_score"
    TEMPORAL_SCORE = "temporal_score"
    POPULARITY_SCORE = "popularity_score"
    COMBINED_SCORE = "combined_score"


class AggregationMethod(Enum):
    """Methods for aggregating results from multiple sources"""

    ROUND_ROBIN = "round_robin"
    SCORE_BASED = "score_based"
    MODALITY_BALANCED = "modality_balanced"
    DIVERSITY_MAXIMIZED = "diversity_maximized"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system"""

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ranking_method: RankingMethod = RankingMethod.COMBINED_SCORE
    aggregation_method: AggregationMethod = AggregationMethod.SCORE_BASED
    max_results: int = 50
    similarity_threshold: float = 0.4
    diversity_threshold: float = 0.8
    temporal_decay_factor: float = 0.1
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    cache_results: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class UserProfile:
    """User preferences and history for personalized retrieval"""

    user_id: str
    preferred_modalities: List[ModalityType] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    preference_weights: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetrievalContext:
    """Context information for retrieval"""

    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    previous_queries: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    location_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Enhanced result with retrieval metadata"""

    search_result: SearchResult
    retrieval_score: float
    ranking_factors: Dict[str, float] = field(default_factory=dict)
    source_strategy: str = ""
    aggregation_rank: int = 0
    diversity_score: float = 0.0
    retrieved_at: datetime = field(default_factory=datetime.utcnow)


class QueryExpander:
    """Expands queries with related terms and concepts"""

    def __init__(self):
        # Simple synonym/related terms mapping
        self.expansion_map = {
            "dog": ["canine", "puppy", "pet", "animal"],
            "cat": ["feline", "kitten", "pet", "animal"],
            "car": ["vehicle", "automobile", "transportation"],
            "music": ["song", "audio", "sound", "melody"],
            "photo": ["image", "picture", "photography"],
            "video": ["movie", "film", "recording", "visual"],
        }

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with related terms"""
        words = query.lower().split()
        expanded_terms = set(words)

        for word in words:
            if word in self.expansion_map:
                related_terms = self.expansion_map[word][:max_expansions]
                expanded_terms.update(related_terms)

        return list(expanded_terms)


class ResultReranker:
    """Re-ranks results based on additional factors"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def rerank_results(
        self, results: List[RetrievalResult], context: Optional[RetrievalContext] = None
    ) -> List[RetrievalResult]:
        """Re-rank results using multiple factors"""

        for result in results:
            # Calculate ranking factors
            factors = self._calculate_ranking_factors(result, context)
            result.ranking_factors = factors

            # Update retrieval score
            result.retrieval_score = self._combine_scores(
                result.search_result.similarity_score, factors
            )

        # Sort by new retrieval score
        results.sort(key=lambda x: x.retrieval_score, reverse=True)

        # Update aggregation ranks
        for i, result in enumerate(results):
            result.aggregation_rank = i + 1

        return results

    def _calculate_ranking_factors(
        self, result: RetrievalResult, context: Optional[RetrievalContext]
    ) -> Dict[str, float]:
        """Calculate various ranking factors"""
        factors = {}

        # Temporal factor (newer content scores higher)
        if hasattr(result.search_result.item, "created_at"):
            age_days = (datetime.utcnow() - result.search_result.item.created_at).days
            temporal_score = max(
                0, 1.0 - age_days * self.config.temporal_decay_factor / 365
            )
            factors["temporal"] = temporal_score
        else:
            factors["temporal"] = 0.5  # Neutral score

        # Modality preference factor
        if context and context.user_profile:
            preferred_modalities = context.user_profile.preferred_modalities
            if result.search_result.item.modality in preferred_modalities:
                factors["modality_preference"] = 1.0
            else:
                factors["modality_preference"] = 0.5
        else:
            factors["modality_preference"] = 0.5

        # Content quality factor (simplified)
        metadata = result.search_result.item.metadata
        quality_indicators = [
            "resolution" in metadata,
            "duration" in metadata,
            "file_size" in metadata,
            len(metadata) > 2,
        ]
        factors["quality"] = sum(quality_indicators) / len(quality_indicators)

        # Diversity factor (how different this result is from others)
        factors["diversity"] = result.diversity_score

        # Feature match strength
        match_count = len(result.search_result.matched_features)
        factors["feature_match"] = min(1.0, match_count / 5.0)

        return factors

    def _combine_scores(
        self, similarity_score: float, factors: Dict[str, float]
    ) -> float:
        """Combine similarity score with ranking factors"""
        if self.config.ranking_method == RankingMethod.SIMILARITY_SCORE:
            return similarity_score

        elif self.config.ranking_method == RankingMethod.COMBINED_SCORE:
            # Weighted combination
            weights = {
                "similarity": 0.4,
                "temporal": 0.15,
                "modality_preference": 0.15,
                "quality": 0.15,
                "diversity": 0.1,
                "feature_match": 0.05,
            }

            combined_score = similarity_score * weights["similarity"]

            for factor, score in factors.items():
                weight = weights.get(factor, 0.0)
                combined_score += score * weight

            return min(1.0, combined_score)

        else:
            return similarity_score


class ResultAggregator:
    """Aggregates results from multiple retrieval strategies"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def aggregate_results(
        self, result_sets: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Aggregate results from multiple sources"""

        if self.config.aggregation_method == AggregationMethod.SCORE_BASED:
            return self._score_based_aggregation(result_sets)

        elif self.config.aggregation_method == AggregationMethod.ROUND_ROBIN:
            return self._round_robin_aggregation(result_sets)

        elif self.config.aggregation_method == AggregationMethod.MODALITY_BALANCED:
            return self._modality_balanced_aggregation(result_sets)

        elif self.config.aggregation_method == AggregationMethod.DIVERSITY_MAXIMIZED:
            return self._diversity_maximized_aggregation(result_sets)

        else:
            # Default to score-based
            return self._score_based_aggregation(result_sets)

    def _score_based_aggregation(
        self, result_sets: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Aggregate based on retrieval scores"""
        all_results = []
        seen_items = set()

        for strategy_name, results in result_sets.items():
            for result in results:
                item_id = result.search_result.item.id
                if item_id not in seen_items:
                    result.source_strategy = strategy_name
                    all_results.append(result)
                    seen_items.add(item_id)

        # Sort by retrieval score
        all_results.sort(key=lambda x: x.retrieval_score, reverse=True)
        return all_results[: self.config.max_results]

    def _round_robin_aggregation(
        self, result_sets: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Round-robin aggregation across strategies"""
        aggregated = []
        seen_items = set()

        # Create iterators for each result set
        iterators = {name: iter(results) for name, results in result_sets.items()}

        # Round-robin selection
        while len(aggregated) < self.config.max_results and iterators:
            for strategy_name in list(iterators.keys()):
                try:
                    result = next(iterators[strategy_name])
                    item_id = result.search_result.item.id

                    if item_id not in seen_items:
                        result.source_strategy = strategy_name
                        aggregated.append(result)
                        seen_items.add(item_id)

                        if len(aggregated) >= self.config.max_results:
                            break

                except StopIteration:
                    # Remove exhausted iterator
                    del iterators[strategy_name]

        return aggregated

    def _modality_balanced_aggregation(
        self, result_sets: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Ensure balanced representation across modalities"""
        all_results = []
        seen_items = set()

        # Collect all unique results
        for strategy_name, results in result_sets.items():
            for result in results:
                item_id = result.search_result.item.id
                if item_id not in seen_items:
                    result.source_strategy = strategy_name
                    all_results.append(result)
                    seen_items.add(item_id)

        # Group by modality
        modality_groups = defaultdict(list)
        for result in all_results:
            modality = result.search_result.item.modality
            modality_groups[modality].append(result)

        # Sort each group by score
        for modality in modality_groups:
            modality_groups[modality].sort(
                key=lambda x: x.retrieval_score, reverse=True
            )

        # Balance across modalities
        balanced_results = []
        max_per_modality = max(1, self.config.max_results // len(modality_groups))

        # First pass: equal distribution
        for modality, results in modality_groups.items():
            balanced_results.extend(results[:max_per_modality])

        # Second pass: fill remaining slots with best scores
        remaining_slots = self.config.max_results - len(balanced_results)
        if remaining_slots > 0:
            remaining_results = []
            for modality, results in modality_groups.items():
                remaining_results.extend(results[max_per_modality:])

            remaining_results.sort(key=lambda x: x.retrieval_score, reverse=True)
            balanced_results.extend(remaining_results[:remaining_slots])

        return balanced_results

    def _diversity_maximized_aggregation(
        self, result_sets: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Maximize diversity while maintaining relevance"""
        all_results = []
        seen_items = set()

        # Collect all unique results
        for strategy_name, results in result_sets.items():
            for result in results:
                item_id = result.search_result.item.id
                if item_id not in seen_items:
                    result.source_strategy = strategy_name
                    all_results.append(result)
                    seen_items.add(item_id)

        # Calculate diversity scores
        self._calculate_diversity_scores(all_results)

        # Greedy selection for diversity
        selected = []
        remaining = all_results.copy()

        # Always select the highest scoring result first
        if remaining:
            best_result = max(remaining, key=lambda x: x.retrieval_score)
            selected.append(best_result)
            remaining.remove(best_result)

        # Select remaining results to maximize diversity
        while len(selected) < self.config.max_results and remaining:
            best_candidate = None
            best_score = -1

            for candidate in remaining:
                # Combined score: relevance + diversity
                diversity_bonus = self._calculate_diversity_bonus(candidate, selected)
                combined_score = 0.7 * candidate.retrieval_score + 0.3 * diversity_bonus

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _calculate_diversity_scores(self, results: List[RetrievalResult]):
        """Calculate diversity scores for results"""
        for i, result in enumerate(results):
            diversity_score = 0.0

            for j, other_result in enumerate(results):
                if i != j:
                    # Calculate similarity between results
                    similarity = self._calculate_result_similarity(result, other_result)
                    diversity_score += 1.0 - similarity

            if len(results) > 1:
                diversity_score /= len(results) - 1

            result.diversity_score = diversity_score

    def _calculate_diversity_bonus(
        self, candidate: RetrievalResult, selected: List[RetrievalResult]
    ) -> float:
        """Calculate diversity bonus for a candidate result"""
        if not selected:
            return 1.0

        min_similarity = 1.0
        for selected_result in selected:
            similarity = self._calculate_result_similarity(candidate, selected_result)
            min_similarity = min(min_similarity, similarity)

        return 1.0 - min_similarity

    def _calculate_result_similarity(
        self, result1: RetrievalResult, result2: RetrievalResult
    ) -> float:
        """Calculate similarity between two results"""
        # Simple similarity based on modality and features
        similarity = 0.0

        # Modality similarity
        if result1.search_result.item.modality == result2.search_result.item.modality:
            similarity += 0.3

        # Feature similarity
        features1 = set(result1.search_result.matched_features)
        features2 = set(result2.search_result.matched_features)

        if features1 or features2:
            feature_intersection = len(features1 & features2)
            feature_union = len(features1 | features2)
            if feature_union > 0:
                similarity += 0.4 * (feature_intersection / feature_union)

        # Content similarity (simplified)
        if (
            result1.search_result.item.text_content
            and result2.search_result.item.text_content
        ):

            words1 = set(result1.search_result.item.text_content.lower().split())
            words2 = set(result2.search_result.item.text_content.lower().split())

            if words1 or words2:
                word_intersection = len(words1 & words2)
                word_union = len(words1 | words2)
                if word_union > 0:
                    similarity += 0.3 * (word_intersection / word_union)

        return min(1.0, similarity)


class UnifiedRetrieval:
    """
    Unified retrieval system that provides seamless access to multi-modal content
    through intelligent search, ranking, and aggregation strategies.
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

        # Initialize components
        embedding_config = EmbeddingConfig(
            embedding_dim=512,
            model_type=EmbeddingModel.SIMPLE,
            normalize_embeddings=True,
        )

        self.search_engine = CrossModalSearchEngine(embedding_dim=512)
        self.embeddings = MultiModalEmbeddings(embedding_config)
        self.query_expander = QueryExpander()
        self.reranker = ResultReranker(config)
        self.aggregator = ResultAggregator(config)

        # User profiles and caching
        self.user_profiles: Dict[str, UserProfile] = {}
        self.result_cache: Dict[str, Tuple[List[RetrievalResult], datetime]] = {}

        logger.info("Initialized UnifiedRetrieval system")

    async def retrieve(
        self,
        query: str,
        query_modality: ModalityType = ModalityType.TEXT,
        target_modalities: Optional[List[ModalityType]] = None,
        context: Optional[RetrievalContext] = None,
        max_results: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Main retrieval function with intelligent strategy selection

        Args:
            query: Search query
            query_modality: Modality of the query
            target_modalities: Target modalities to search
            context: Retrieval context for personalization
            max_results: Maximum number of results

        Returns:
            List of ranked retrieval results
        """
        max_results = max_results or self.config.max_results

        # Check cache first
        cache_key = self._generate_cache_key(
            query, query_modality, target_modalities, context
        )
        if self.config.cache_results:
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                logger.debug(f"Retrieved {len(cached_results)} cached results")
                return cached_results[:max_results]

        # Expand query if enabled
        expanded_queries = [query]
        if self.config.enable_query_expansion:
            expanded_terms = self.query_expander.expand_query(query)
            expanded_queries.extend(expanded_terms)

        # Execute retrieval strategies
        result_sets = {}

        if self.config.strategy in [
            RetrievalStrategy.SIMILARITY_ONLY,
            RetrievalStrategy.HYBRID,
        ]:
            result_sets["similarity"] = await self._similarity_retrieval(
                expanded_queries, query_modality, target_modalities, max_results
            )

        if self.config.strategy in [
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.CONTEXTUAL,
        ]:
            result_sets["contextual"] = await self._contextual_retrieval(
                expanded_queries,
                query_modality,
                target_modalities,
                context,
                max_results,
            )

        if (
            self.config.strategy == RetrievalStrategy.PERSONALIZED
            and context
            and context.user_profile
        ):
            result_sets["personalized"] = await self._personalized_retrieval(
                expanded_queries,
                query_modality,
                target_modalities,
                context,
                max_results,
            )

        if self.config.strategy == RetrievalStrategy.MULTI_STAGE:
            result_sets.update(
                await self._multi_stage_retrieval(
                    expanded_queries,
                    query_modality,
                    target_modalities,
                    context,
                    max_results,
                )
            )

        # Aggregate results from different strategies
        if not result_sets:
            # Fallback to basic similarity search
            result_sets["fallback"] = await self._similarity_retrieval(
                [query], query_modality, target_modalities, max_results
            )

        aggregated_results = self.aggregator.aggregate_results(result_sets)

        # Re-rank if enabled
        if self.config.enable_reranking:
            aggregated_results = self.reranker.rerank_results(
                aggregated_results, context
            )

        # Limit results
        final_results = aggregated_results[:max_results]

        # Cache results
        if self.config.cache_results:
            self._cache_results(cache_key, final_results)

        # Update user profile if provided
        if context and context.user_profile:
            self._update_user_profile(context.user_profile, query, final_results)

        logger.info(
            f"Retrieved {len(final_results)} results for query: {query[:50]}..."
        )
        return final_results

    async def _similarity_retrieval(
        self,
        queries: List[str],
        query_modality: ModalityType,
        target_modalities: Optional[List[ModalityType]],
        max_results: int,
    ) -> List[RetrievalResult]:
        """Basic similarity-based retrieval"""
        all_results = []

        for query_text in queries:
            search_query = SearchQuery(
                query=query_text,
                query_modality=query_modality,
                target_modalities=target_modalities or list(ModalityType),
                max_results=max_results,
                threshold=self.config.similarity_threshold,
            )

            search_results = await self.search_engine.search(search_query)

            # Convert to RetrievalResults
            for search_result in search_results:
                retrieval_result = RetrievalResult(
                    search_result=search_result,
                    retrieval_score=search_result.similarity_score,
                    source_strategy="similarity",
                )
                all_results.append(retrieval_result)

        # Remove duplicates and sort
        seen_ids = set()
        unique_results = []
        for result in all_results:
            item_id = result.search_result.item.id
            if item_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(item_id)

        unique_results.sort(key=lambda x: x.retrieval_score, reverse=True)
        return unique_results[:max_results]

    async def _contextual_retrieval(
        self,
        queries: List[str],
        query_modality: ModalityType,
        target_modalities: Optional[List[ModalityType]],
        context: Optional[RetrievalContext],
        max_results: int,
    ) -> List[RetrievalResult]:
        """Context-aware retrieval"""
        # For now, similar to similarity retrieval but with context-based filtering
        results = await self._similarity_retrieval(
            queries, query_modality, target_modalities, max_results
        )

        # Apply contextual modifications
        if context:
            # Boost recent items if in temporal context
            if context.temporal_context.get("prefer_recent", False):
                for result in results:
                    if hasattr(result.search_result.item, "created_at"):
                        age_days = (
                            datetime.utcnow() - result.search_result.item.created_at
                        ).days
                        if age_days < 7:  # Recent items get boost
                            result.retrieval_score *= 1.2

            # Consider previous queries for context
            if context.previous_queries:
                for result in results:
                    # Simple context matching
                    for prev_query in context.previous_queries[-3:]:  # Last 3 queries
                        if result.search_result.item.text_content:
                            if any(
                                word in result.search_result.item.text_content.lower()
                                for word in prev_query.lower().split()
                            ):
                                result.retrieval_score *= 1.1
                                break

        results.sort(key=lambda x: x.retrieval_score, reverse=True)
        return results

    async def _personalized_retrieval(
        self,
        queries: List[str],
        query_modality: ModalityType,
        target_modalities: Optional[List[ModalityType]],
        context: RetrievalContext,
        max_results: int,
    ) -> List[RetrievalResult]:
        """Personalized retrieval based on user profile"""
        user_profile = context.user_profile

        # Adjust target modalities based on user preferences
        if user_profile.preferred_modalities:
            if target_modalities:
                # Prioritize preferred modalities
                preferred_target = [
                    m
                    for m in target_modalities
                    if m in user_profile.preferred_modalities
                ]
                other_target = [
                    m
                    for m in target_modalities
                    if m not in user_profile.preferred_modalities
                ]
                target_modalities = preferred_target + other_target
            else:
                target_modalities = user_profile.preferred_modalities

        # Get base results
        results = await self._similarity_retrieval(
            queries, query_modality, target_modalities, max_results
        )

        # Apply personalization scoring
        for result in results:
            personalization_score = 1.0

            # Modality preference
            if result.search_result.item.modality in user_profile.preferred_modalities:
                personalization_score *= 1.3

            # Historical interaction patterns
            for interaction in user_profile.interaction_history[
                -10:
            ]:  # Recent interactions
                if interaction.get("item_id") == result.search_result.item.id:
                    personalization_score *= 1.5  # Previously interacted
                    break

            # Preference weights
            for key, weight in user_profile.preference_weights.items():
                if key in result.search_result.item.metadata:
                    personalization_score *= 1.0 + weight * 0.2

            result.retrieval_score *= personalization_score

        results.sort(key=lambda x: x.retrieval_score, reverse=True)
        return results

    async def _multi_stage_retrieval(
        self,
        queries: List[str],
        query_modality: ModalityType,
        target_modalities: Optional[List[ModalityType]],
        context: Optional[RetrievalContext],
        max_results: int,
    ) -> Dict[str, List[RetrievalResult]]:
        """Multi-stage retrieval pipeline"""
        result_sets = {}

        # Stage 1: Broad similarity search
        stage1_results = await self._similarity_retrieval(
            queries, query_modality, target_modalities, max_results * 2
        )
        result_sets["stage1_similarity"] = stage1_results

        # Stage 2: Context-aware refinement
        if context:
            stage2_results = await self._contextual_retrieval(
                queries, query_modality, target_modalities, context, max_results
            )
            result_sets["stage2_contextual"] = stage2_results

        # Stage 3: Personalized results (if user profile available)
        if context and context.user_profile:
            stage3_results = await self._personalized_retrieval(
                queries, query_modality, target_modalities, context, max_results
            )
            result_sets["stage3_personalized"] = stage3_results

        return result_sets

    def create_user_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile"""
        profile = UserProfile(user_id=user_id)
        self.user_profiles[user_id] = profile
        return profile

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.user_profiles.get(user_id)

    def _update_user_profile(
        self, profile: UserProfile, query: str, results: List[RetrievalResult]
    ):
        """Update user profile based on search interaction"""
        # Add to interaction history
        interaction = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "result_count": len(results),
            "top_modalities": [
                r.search_result.item.modality.value for r in results[:5]
            ],
        }
        profile.interaction_history.append(interaction)

        # Limit history size
        if len(profile.interaction_history) > 100:
            profile.interaction_history = profile.interaction_history[-100:]

        # Update modality preferences
        modality_counts = Counter()
        for result in results[:10]:  # Top 10 results
            modality_counts[result.search_result.item.modality] += 1

        # Update preferred modalities
        top_modalities = [modality for modality, _ in modality_counts.most_common(3)]
        profile.preferred_modalities = top_modalities

        profile.last_updated = datetime.utcnow()

    def _generate_cache_key(
        self,
        query: str,
        query_modality: ModalityType,
        target_modalities: Optional[List[ModalityType]],
        context: Optional[RetrievalContext],
    ) -> str:
        """Generate cache key for query"""
        key_parts = [
            query,
            query_modality.value,
            (
                ",".join(sorted([m.value for m in target_modalities]))
                if target_modalities
                else "all"
            ),
            str(self.config.strategy.value),
            str(self.config.max_results),
        ]

        if context and context.user_profile:
            key_parts.append(context.user_profile.user_id)

        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _get_cached_results(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Get cached results if valid"""
        if cache_key in self.result_cache:
            results, cached_at = self.result_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(
                seconds=self.config.cache_ttl_seconds
            ):
                return results
            else:
                # Remove expired cache entry
                del self.result_cache[cache_key]
        return None

    def _cache_results(self, cache_key: str, results: List[RetrievalResult]):
        """Cache results"""
        self.result_cache[cache_key] = (results, datetime.utcnow())

        # Limit cache size
        if len(self.result_cache) > 1000:
            # Remove oldest 25% of entries
            sorted_items = sorted(self.result_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:250]:
                del self.result_cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "search_engine": self.search_engine.get_stats(),
            "embeddings": self.embeddings.get_statistics(),
            "user_profiles": len(self.user_profiles),
            "cache_entries": len(self.result_cache),
            "config": {
                "strategy": self.config.strategy.value,
                "ranking_method": self.config.ranking_method.value,
                "aggregation_method": self.config.aggregation_method.value,
                "max_results": self.config.max_results,
            },
        }

        return stats

    def clear_cache(self):
        """Clear result cache"""
        self.result_cache.clear()
        self.search_engine.clear_cache()
        logger.info("Cleared retrieval cache")

    async def index_content(self, content_item: ContentItem) -> bool:
        """Index new content for retrieval"""
        # Index in search engine
        search_success = await self.search_engine.index_content(content_item)

        # Create embeddings
        embedding_success = False
        try:
            if content_item.modality == ModalityType.TEXT:
                await self.embeddings.embed_text(
                    content_item.text_content or "",
                    content_item.id,
                    content_item.metadata,
                )
            elif content_item.modality == ModalityType.IMAGE:
                await self.embeddings.embed_image(
                    content_item.content_path, content_item.id, content_item.metadata
                )
            elif content_item.modality == ModalityType.AUDIO:
                await self.embeddings.embed_audio(
                    content_item.content_path, content_item.id, content_item.metadata
                )
            elif content_item.modality == ModalityType.VIDEO:
                await self.embeddings.embed_video(
                    content_item.content_path, content_item.id, content_item.metadata
                )
            elif content_item.modality == ModalityType.DOCUMENT:
                await self.embeddings.embed_document(
                    content_item.content_path, content_item.id, content_item.metadata
                )

            embedding_success = True

        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")

        return search_success and embedding_success
