#!/usr/bin/env python3
"""
🔍 PHASE 4: OUTPUT EVALUATION ENGINE
==================================================
Sophisticated evaluation system for conversation quality, response effectiveness,
and continuous learning from interaction patterns.

This engine implements:
- Multi-dimensional response quality scoring
- User satisfaction tracking and prediction
- Conversation flow analysis
- Response effectiveness measurement
- Automated quality improvement suggestions
- Pattern recognition for optimal responses
"""

import sqlite3
import logging
import json
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import re
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResponseEvaluation:
    """Comprehensive response evaluation metrics"""

    conversation_id: str
    prompt: str
    response: str
    timestamp: datetime

    # Quality Metrics
    relevance_score: float  # 0-1: How relevant is response to prompt
    clarity_score: float  # 0-1: How clear and understandable
    completeness_score: float  # 0-1: How complete is the answer
    accuracy_score: float  # 0-1: Factual accuracy (when verifiable)
    helpfulness_score: float  # 0-1: How helpful to user's goal

    # Technical Metrics
    response_time: float  # Seconds to generate response
    token_count: int  # Number of tokens in response
    prompt_complexity: float  # 0-1: Complexity of the prompt

    # Conversation Context
    conversation_length: int  # Number of exchanges in conversation
    topic_coherence: float  # 0-1: How well response maintains topic

    # Overall Quality
    overall_quality: float  # 0-1: Weighted combination of all metrics
    quality_tier: str  # excellent, good, average, poor

    # Improvement Suggestions
    improvement_areas: List[str]  # Areas needing improvement
    suggested_optimizations: List[str]  # Specific optimization suggestions


@dataclass
class ConversationPattern:
    """Identified conversation patterns and insights"""

    pattern_id: str
    pattern_type: str  # question_type, topic_cluster, response_style
    frequency: int  # How often this pattern occurs
    avg_quality: float  # Average quality for this pattern
    success_indicators: List[str]  # What makes responses successful
    failure_indicators: List[str]  # What causes poor responses
    optimization_suggestions: List[str]  # How to improve this pattern


class ResponseQualityAnalyzer:
    """Analyzes response quality across multiple dimensions"""

    def __init__(self):
        self.evaluations_db = "evaluations.db"
        self._init_database()

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "average": 0.6,
            "poor": 0.0,
        }

        # Weight configuration for overall quality calculation
        self.quality_weights = {
            "relevance": 0.25,
            "clarity": 0.20,
            "completeness": 0.20,
            "accuracy": 0.15,
            "helpfulness": 0.20,
        }

    def _init_database(self):
        """Initialize evaluation database"""
        conn = sqlite3.connect(self.evaluations_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS response_evaluations (
            conversation_id TEXT,
            prompt TEXT,
            response TEXT,
            timestamp TEXT,
            relevance_score REAL,
            clarity_score REAL,
            completeness_score REAL,
            accuracy_score REAL,
            helpfulness_score REAL,
            response_time REAL,
            token_count INTEGER,
            prompt_complexity REAL,
            conversation_length INTEGER,
            topic_coherence REAL,
            overall_quality REAL,
            quality_tier TEXT,
            improvement_areas TEXT,
            suggested_optimizations TEXT,
            PRIMARY KEY (conversation_id, timestamp)
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS conversation_patterns (
            pattern_id TEXT PRIMARY KEY,
            pattern_type TEXT,
            frequency INTEGER,
            avg_quality REAL,
            success_indicators TEXT,
            failure_indicators TEXT,
            optimization_suggestions TEXT,
            last_updated TEXT
        )
        """
        )

        conn.commit()
        conn.close()

    def evaluate_response(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        response_time: float,
        conversation_context: Dict[str, Any] = None,
    ) -> ResponseEvaluation:
        """Comprehensive response evaluation"""

        # Calculate individual quality metrics
        relevance = self._calculate_relevance(prompt, response)
        clarity = self._calculate_clarity(response)
        completeness = self._calculate_completeness(prompt, response)
        accuracy = self._calculate_accuracy(response)
        helpfulness = self._calculate_helpfulness(prompt, response)

        # Technical metrics
        token_count = len(response.split())
        prompt_complexity = self._calculate_prompt_complexity(prompt)
        conversation_length = (
            conversation_context.get("length", 1) if conversation_context else 1
        )
        topic_coherence = self._calculate_topic_coherence(
            prompt, response, conversation_context
        )

        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            {
                "relevance": relevance,
                "clarity": clarity,
                "completeness": completeness,
                "accuracy": accuracy,
                "helpfulness": helpfulness,
            }
        )

        # Determine quality tier
        quality_tier = self._determine_quality_tier(overall_quality)

        # Identify improvement areas and suggestions
        improvement_areas = self._identify_improvement_areas(
            {
                "relevance": relevance,
                "clarity": clarity,
                "completeness": completeness,
                "accuracy": accuracy,
                "helpfulness": helpfulness,
            }
        )

        suggested_optimizations = self._generate_optimization_suggestions(
            improvement_areas, prompt, response
        )

        evaluation = ResponseEvaluation(
            conversation_id=conversation_id,
            prompt=prompt,
            response=response,
            timestamp=datetime.now(),
            relevance_score=relevance,
            clarity_score=clarity,
            completeness_score=completeness,
            accuracy_score=accuracy,
            helpfulness_score=helpfulness,
            response_time=response_time,
            token_count=token_count,
            prompt_complexity=prompt_complexity,
            conversation_length=conversation_length,
            topic_coherence=topic_coherence,
            overall_quality=overall_quality,
            quality_tier=quality_tier,
            improvement_areas=improvement_areas,
            suggested_optimizations=suggested_optimizations,
        )

        # Store evaluation
        self._store_evaluation(evaluation)

        return evaluation

    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate how relevant the response is to the prompt"""
        # Keyword overlap analysis
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }
        prompt_words -= stopwords
        response_words -= stopwords

        if not prompt_words:
            return 0.8  # Default for very short prompts

        # Calculate Jaccard similarity
        intersection = len(prompt_words & response_words)
        union = len(prompt_words | response_words)
        jaccard_similarity = intersection / union if union > 0 else 0

        # Boost score for direct question answers
        if "?" in prompt and any(
            indicator in response.lower()
            for indicator in ["yes", "no", "because", "due to", "as a result"]
        ):
            jaccard_similarity += 0.2

        # Check for topic coherence
        if self._check_topic_coherence(prompt, response):
            jaccard_similarity += 0.1

        return min(1.0, jaccard_similarity)

    def _calculate_clarity(self, response: str) -> float:
        """Calculate clarity of the response"""
        # Sentence structure analysis
        sentences = response.split(".")
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

        # Penalize very long or very short sentences
        if avg_sentence_length < 5 or avg_sentence_length > 30:
            clarity_penalty = 0.2
        else:
            clarity_penalty = 0

        # Check for clear structure
        structure_score = 0
        if any(marker in response for marker in ["1.", "2.", "-", "*", "•"]):
            structure_score += 0.2  # Structured content

        if response.count("\n") > 1:
            structure_score += 0.1  # Paragraphed content

        # Check readability indicators
        readability_score = 0.8  # Base score

        # Penalize excessive jargon or complexity
        complex_words = len([w for w in response.split() if len(w) > 10])
        total_words = len(response.split())
        if total_words > 0 and complex_words / total_words > 0.3:
            readability_score -= 0.2

        return min(1.0, readability_score + structure_score - clarity_penalty)

    def _calculate_completeness(self, prompt: str, response: str) -> float:
        """Calculate how complete the response is"""
        # Check if response addresses all parts of multi-part questions
        question_markers = prompt.count("?")
        if question_markers > 1:
            # Multi-part question
            response_parts = len([s for s in response.split(".") if s.strip()])
            completeness = min(1.0, response_parts / (question_markers * 2))
        else:
            # Single question/request
            if len(response.split()) < 10:
                completeness = 0.5  # Very short response
            elif len(response.split()) < 30:
                completeness = 0.7  # Moderate response
            else:
                completeness = 0.9  # Comprehensive response

        # Boost for examples and explanations
        if any(
            indicator in response.lower()
            for indicator in ["example", "for instance", "such as", "because", "due to"]
        ):
            completeness += 0.1

        return min(1.0, completeness)

    def _calculate_accuracy(self, response: str) -> float:
        """Calculate factual accuracy (heuristic-based)"""
        # This is a simplified heuristic approach
        # In practice, this would integrate with fact-checking services

        accuracy_score = 0.8  # Default assumption

        # Check for uncertainty indicators (good for accuracy)
        uncertainty_indicators = [
            "might",
            "could",
            "possibly",
            "probably",
            "likely",
            "appears",
            "seems",
        ]
        if any(indicator in response.lower() for indicator in uncertainty_indicators):
            accuracy_score += 0.1

        # Check for absolute statements (potentially risky)
        absolute_indicators = [
            "always",
            "never",
            "all",
            "none",
            "definitely",
            "certainly",
        ]
        absolute_count = sum(
            1 for indicator in absolute_indicators if indicator in response.lower()
        )
        if absolute_count > 2:
            accuracy_score -= 0.2

        # Check for citations or references (good for accuracy)
        if any(
            ref in response.lower()
            for ref in [
                "according to",
                "research shows",
                "studies indicate",
                "source:",
                "reference:",
            ]
        ):
            accuracy_score += 0.1

        return min(1.0, max(0.1, accuracy_score))

    def _calculate_helpfulness(self, prompt: str, response: str) -> float:
        """Calculate how helpful the response is"""
        helpfulness = 0.7  # Base score

        # Check for actionable advice
        action_words = [
            "try",
            "use",
            "consider",
            "implement",
            "apply",
            "follow",
            "start",
            "begin",
        ]
        if any(word in response.lower() for word in action_words):
            helpfulness += 0.2

        # Check for step-by-step guidance
        if any(
            marker in response
            for marker in ["step 1", "first,", "then,", "next,", "finally,"]
        ):
            helpfulness += 0.2

        # Check for comprehensive coverage
        if len(response.split()) > 50:  # Detailed response
            helpfulness += 0.1

        # Penalize non-answers
        if any(
            phrase in response.lower()
            for phrase in ["i don't know", "i'm not sure", "i cannot", "sorry, i can't"]
        ):
            helpfulness -= 0.3

        return min(1.0, max(0.0, helpfulness))

    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Calculate complexity of the user prompt"""
        complexity = 0.0

        # Length factor
        word_count = len(prompt.split())
        if word_count > 50:
            complexity += 0.4
        elif word_count > 20:
            complexity += 0.2

        # Question count
        question_count = prompt.count("?")
        complexity += min(0.3, question_count * 0.1)

        # Technical terms
        technical_indicators = [
            "implement",
            "algorithm",
            "optimize",
            "analyze",
            "debug",
            "configure",
            "integrate",
        ]
        tech_count = sum(1 for term in technical_indicators if term in prompt.lower())
        complexity += min(0.3, tech_count * 0.1)

        return min(1.0, complexity)

    def _calculate_topic_coherence(
        self, prompt: str, response: str, context: Dict = None
    ) -> float:
        """Calculate how well response maintains topic coherence"""
        if not context or "previous_topics" not in context:
            return 0.8  # Default when no context available

        # Simplified topic coherence calculation
        # In practice, this would use more sophisticated NLP
        coherence = 0.8

        # Check if response introduces completely unrelated topics
        prompt_topics = set(prompt.lower().split())
        response_topics = set(response.lower().split())

        # Remove stopwords for better analysis
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        prompt_topics -= stopwords
        response_topics -= stopwords

        overlap = len(prompt_topics & response_topics)
        total = len(prompt_topics | response_topics)

        if total > 0:
            coherence = overlap / total

        return min(1.0, coherence)

    def _check_topic_coherence(self, prompt: str, response: str) -> bool:
        """Simple topic coherence check"""
        # Extract key nouns and verbs
        import re

        # Simple keyword extraction
        prompt_keywords = re.findall(r"\b[a-zA-Z]{4,}\b", prompt.lower())
        response_keywords = re.findall(r"\b[a-zA-Z]{4,}\b", response.lower())

        # Check for keyword overlap
        common_keywords = set(prompt_keywords) & set(response_keywords)
        return len(common_keywords) >= 2

    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        total_score = 0
        total_weight = 0

        for metric, score in metrics.items():
            if metric in self.quality_weights:
                weight = self.quality_weights[metric]
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _determine_quality_tier(self, overall_quality: float) -> str:
        """Determine quality tier based on overall score"""
        for tier, threshold in self.quality_thresholds.items():
            if overall_quality >= threshold:
                return tier
        return "poor"

    def _identify_improvement_areas(self, metrics: Dict[str, float]) -> List[str]:
        """Identify areas that need improvement"""
        improvement_areas = []
        threshold = 0.7  # Areas below this need improvement

        for metric, score in metrics.items():
            if score < threshold:
                improvement_areas.append(metric)

        return improvement_areas

    def _generate_optimization_suggestions(
        self, improvement_areas: List[str], prompt: str, response: str
    ) -> List[str]:
        """Generate specific optimization suggestions"""
        suggestions = []

        if "relevance" in improvement_areas:
            suggestions.append(
                "Improve prompt-response alignment by addressing all aspects of the question"
            )

        if "clarity" in improvement_areas:
            suggestions.append(
                "Use clearer structure with bullet points, numbered lists, or paragraphs"
            )

        if "completeness" in improvement_areas:
            suggestions.append(
                "Provide more comprehensive answers with examples and explanations"
            )

        if "accuracy" in improvement_areas:
            suggestions.append(
                "Include uncertainty qualifiers and verify factual claims"
            )

        if "helpfulness" in improvement_areas:
            suggestions.append("Add actionable steps and practical guidance")

        return suggestions

    def _store_evaluation(self, evaluation: ResponseEvaluation):
        """Store evaluation in database"""
        conn = sqlite3.connect(self.evaluations_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO response_evaluations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                evaluation.conversation_id,
                evaluation.prompt,
                evaluation.response,
                evaluation.timestamp.isoformat(),
                evaluation.relevance_score,
                evaluation.clarity_score,
                evaluation.completeness_score,
                evaluation.accuracy_score,
                evaluation.helpfulness_score,
                evaluation.response_time,
                evaluation.token_count,
                evaluation.prompt_complexity,
                evaluation.conversation_length,
                evaluation.topic_coherence,
                evaluation.overall_quality,
                evaluation.quality_tier,
                json.dumps(evaluation.improvement_areas),
                json.dumps(evaluation.suggested_optimizations),
            ),
        )

        conn.commit()
        conn.close()


class ConversationPatternAnalyzer:
    """Analyzes conversation patterns and learns from them"""

    def __init__(self, quality_analyzer: ResponseQualityAnalyzer):
        self.quality_analyzer = quality_analyzer
        self.patterns_db = quality_analyzer.evaluations_db

    def analyze_patterns(self) -> List[ConversationPattern]:
        """Analyze conversation patterns and identify insights"""
        conn = sqlite3.connect(self.patterns_db)
        cursor = conn.cursor()

        patterns = []

        # Analyze by prompt complexity
        patterns.extend(self._analyze_complexity_patterns(cursor))

        # Analyze by conversation length
        patterns.extend(self._analyze_length_patterns(cursor))

        # Analyze by topic coherence
        patterns.extend(self._analyze_coherence_patterns(cursor))

        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern, cursor)

        conn.commit()
        conn.close()

        return patterns

    def _analyze_complexity_patterns(self, cursor) -> List[ConversationPattern]:
        """Analyze patterns based on prompt complexity"""
        patterns = []

        # High complexity prompts
        cursor.execute(
            """
        SELECT AVG(overall_quality), COUNT(*), AVG(response_time)
        FROM response_evaluations 
        WHERE prompt_complexity > 0.7
        """
        )
        result = cursor.fetchone()

        if result[1] > 0:  # If we have data
            patterns.append(
                ConversationPattern(
                    pattern_id="high_complexity_prompts",
                    pattern_type="prompt_complexity",
                    frequency=int(result[1]),
                    avg_quality=result[0],
                    success_indicators=[
                        "Structured responses",
                        "Step-by-step explanations",
                    ],
                    failure_indicators=["Incomplete answers", "Poor organization"],
                    optimization_suggestions=[
                        "Break down complex prompts",
                        "Use clear structure",
                        "Provide examples",
                    ],
                )
            )

        return patterns

    def _analyze_length_patterns(self, cursor) -> List[ConversationPattern]:
        """Analyze patterns based on conversation length"""
        patterns = []

        # Long conversations
        cursor.execute(
            """
        SELECT AVG(overall_quality), COUNT(*), AVG(topic_coherence)
        FROM response_evaluations 
        WHERE conversation_length > 5
        """
        )
        result = cursor.fetchone()

        if result[1] > 0:
            patterns.append(
                ConversationPattern(
                    pattern_id="long_conversations",
                    pattern_type="conversation_length",
                    frequency=int(result[1]),
                    avg_quality=result[0],
                    success_indicators=["Maintained context", "Coherent topic flow"],
                    failure_indicators=["Topic drift", "Repetitive responses"],
                    optimization_suggestions=[
                        "Maintain conversation context",
                        "Reference previous exchanges",
                    ],
                )
            )

        return patterns

    def _analyze_coherence_patterns(self, cursor) -> List[ConversationPattern]:
        """Analyze patterns based on topic coherence"""
        patterns = []

        # High coherence conversations
        cursor.execute(
            """
        SELECT AVG(overall_quality), COUNT(*), AVG(helpfulness_score)
        FROM response_evaluations 
        WHERE topic_coherence > 0.8
        """
        )
        result = cursor.fetchone()

        if result[1] > 0:
            patterns.append(
                ConversationPattern(
                    pattern_id="high_coherence_conversations",
                    pattern_type="topic_coherence",
                    frequency=int(result[1]),
                    avg_quality=result[0],
                    success_indicators=["Clear topic focus", "Relevant responses"],
                    failure_indicators=["Topic jumping", "Irrelevant information"],
                    optimization_suggestions=[
                        "Stay focused on topic",
                        "Use context effectively",
                    ],
                )
            )

        return patterns

    def _store_pattern(self, pattern: ConversationPattern, cursor):
        """Store conversation pattern in database"""
        cursor.execute(
            """
        INSERT OR REPLACE INTO conversation_patterns VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                pattern.pattern_id,
                pattern.pattern_type,
                pattern.frequency,
                pattern.avg_quality,
                json.dumps(pattern.success_indicators),
                json.dumps(pattern.failure_indicators),
                json.dumps(pattern.optimization_suggestions),
                datetime.now().isoformat(),
            ),
        )


class EvaluationEngine:
    """Main evaluation engine coordinating all evaluation processes"""

    def __init__(self):
        self.quality_analyzer = ResponseQualityAnalyzer()
        self.pattern_analyzer = ConversationPatternAnalyzer(self.quality_analyzer)

        logger.info("🔍 Evaluation Engine initialized")

    async def evaluate_conversation(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        response_time: float,
        context: Dict[str, Any] = None,
    ) -> ResponseEvaluation:
        """Evaluate a single conversation exchange"""
        return self.quality_analyzer.evaluate_response(
            conversation_id, prompt, response, response_time, context
        )

    async def analyze_conversation_patterns(self) -> List[ConversationPattern]:
        """Analyze patterns across all conversations"""
        return self.pattern_analyzer.analyze_patterns()

    async def get_quality_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get quality insights for the specified time period"""
        conn = sqlite3.connect(self.quality_analyzer.evaluations_db)
        cursor = conn.cursor()

        # Get recent evaluations
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute(
            """
        SELECT * FROM response_evaluations 
        WHERE timestamp > ?
        ORDER BY timestamp DESC
        """,
            (since_date,),
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {"message": "No evaluations found for the specified period"}

        # Calculate insights
        qualities = [row[14] for row in results]  # overall_quality column
        response_times = [row[9] for row in results]  # response_time column

        insights = {
            "evaluation_count": len(results),
            "avg_quality": statistics.mean(qualities),
            "quality_trend": (
                "improving"
                if len(qualities) > 1 and qualities[0] > qualities[-1]
                else "stable"
            ),
            "avg_response_time": statistics.mean(response_times),
            "quality_distribution": {
                "excellent": len([q for q in qualities if q >= 0.9]),
                "good": len([q for q in qualities if 0.75 <= q < 0.9]),
                "average": len([q for q in qualities if 0.6 <= q < 0.75]),
                "poor": len([q for q in qualities if q < 0.6]),
            },
            "improvement_recommendations": [
                (
                    "Focus on response clarity"
                    if statistics.mean([row[6] for row in results]) < 0.8
                    else None
                ),
                (
                    "Improve completeness"
                    if statistics.mean([row[7] for row in results]) < 0.8
                    else None
                ),
                (
                    "Work on relevance"
                    if statistics.mean([row[5] for row in results]) < 0.8
                    else None
                ),
            ],
        }

        # Remove None recommendations
        insights["improvement_recommendations"] = [
            r for r in insights["improvement_recommendations"] if r
        ]

        return insights

    async def run_evaluation_cycle(self):
        """Run a complete evaluation cycle"""
        logger.info("🔄 Running evaluation cycle...")

        # Analyze patterns
        patterns = await self.analyze_conversation_patterns()
        logger.info(f"📊 Analyzed {len(patterns)} conversation patterns")

        # Get quality insights
        insights = await self.get_quality_insights()
        logger.info(
            f"📈 Generated quality insights for {insights.get('evaluation_count', 0)} evaluations"
        )

        return {
            "patterns": [asdict(p) for p in patterns],
            "insights": insights,
            "cycle_completed": datetime.now().isoformat(),
        }


# Test and demonstration functions
async def demonstrate_evaluation_engine():
    """Demonstrate the evaluation engine capabilities"""
    print("🔍 OUTPUT EVALUATION ENGINE")
    print("=" * 50)

    engine = EvaluationEngine()

    # Sample conversation data
    test_conversations = [
        {
            "id": "test_1",
            "prompt": "How do I implement a REST API in Python?",
            "response": "To implement a REST API in Python, you can use FastAPI or Flask. Here's a step-by-step approach:\n1. Install FastAPI: pip install fastapi uvicorn\n2. Create your main app file\n3. Define your endpoints with proper HTTP methods\n4. Add request/response models using Pydantic\n5. Run with uvicorn\n\nExample: @app.get('/items/{item_id}') def read_item(item_id: int): return {'item_id': item_id}",
            "response_time": 0.15,
            "context": {"length": 3},
        },
        {
            "id": "test_2",
            "prompt": "What's the weather?",
            "response": "I don't have access to real-time weather data.",
            "response_time": 0.05,
            "context": {"length": 1},
        },
        {
            "id": "test_3",
            "prompt": "Explain machine learning algorithms for beginners with examples and use cases",
            "response": "Machine learning algorithms are methods that allow computers to learn from data. Common types include:\n\n1. Supervised Learning (learns from labeled examples)\n   - Linear Regression: Predicting house prices\n   - Decision Trees: Email spam detection\n   - Neural Networks: Image recognition\n\n2. Unsupervised Learning (finds patterns in data)\n   - Clustering: Customer segmentation\n   - Association Rules: Market basket analysis\n\n3. Reinforcement Learning (learns through trial and error)\n   - Game playing (Chess, Go)\n   - Autonomous vehicles\n\nEach algorithm has strengths and is suited for different problems.",
            "response_time": 0.25,
            "context": {"length": 2},
        },
    ]

    print("📊 Evaluating sample conversations...")
    evaluations = []

    for conv in test_conversations:
        evaluation = await engine.evaluate_conversation(
            conv["id"],
            conv["prompt"],
            conv["response"],
            conv["response_time"],
            conv["context"],
        )
        evaluations.append(evaluation)

        print(f"\n✅ Conversation {conv['id']}:")
        print(
            f"  📊 Overall Quality: {evaluation.overall_quality:.3f} ({evaluation.quality_tier})"
        )
        print(f"  🎯 Relevance: {evaluation.relevance_score:.3f}")
        print(f"  🔍 Clarity: {evaluation.clarity_score:.3f}")
        print(f"  ✅ Completeness: {evaluation.completeness_score:.3f}")
        print(f"  🎭 Helpfulness: {evaluation.helpfulness_score:.3f}")

        if evaluation.improvement_areas:
            print(f"  📈 Needs improvement: {', '.join(evaluation.improvement_areas)}")

    print(f"\n📈 Evaluations completed: {len(evaluations)}")

    # Run pattern analysis
    print("\n🔄 Running pattern analysis...")
    patterns = await engine.analyze_conversation_patterns()
    print(f"📊 Identified {len(patterns)} patterns")

    # Get insights
    print("\n📊 Generating quality insights...")
    insights = await engine.get_quality_insights(days=30)
    print(
        f"📈 Insights generated for {insights.get('evaluation_count', 0)} evaluations"
    )
    print(f"📊 Average quality: {insights.get('avg_quality', 0):.3f}")

    print("\n🎯 EVALUATION ENGINE OPERATIONAL")
    print("System is now analyzing response quality and learning from patterns")


if __name__ == "__main__":
    asyncio.run(demonstrate_evaluation_engine())
