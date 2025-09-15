"""
Enhanced Conversation Integration for Vega2.0

This module integrates the autonomous AI improvement system directly with
chat conversations, providing:
- Real-time improvement suggestions during conversations
- Quality analysis of ongoing conversations
- Automatic improvement triggers based on conversation patterns
- User feedback integration with the improvement system
- Conversation-driven skill development

Integration Points:
1. Chat endpoint enhancement with improvement tracking
2. Real-time conversation quality analysis
3. Improvement suggestion generation
4. User feedback collection and processing
5. Conversation pattern recognition for auto-improvements
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ConversationImprovement:
    """Suggested improvement for ongoing conversation"""

    conversation_id: str
    improvement_type: str
    suggestion: str
    confidence: float
    estimated_impact: float
    auto_apply: bool
    context: Dict[str, Any]


@dataclass
class ConversationQuality:
    """Real-time conversation quality metrics"""

    conversation_id: str
    relevance_score: float
    clarity_score: float
    helpfulness_score: float
    completeness_score: float
    overall_score: float
    improvement_areas: List[str]
    timestamp: str


class ConversationIntegrator:
    """Integrates autonomous improvements with chat conversations"""

    def __init__(self):
        self.active_conversations = {}
        self.improvement_suggestions = {}
        self.quality_thresholds = {"excellent": 0.85, "good": 0.70, "poor": 0.50}

    async def analyze_conversation_quality(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        session_id: Optional[str] = None,
    ) -> ConversationQuality:
        """Analyze the quality of a conversation exchange"""

        # Basic quality scoring (can be enhanced with LLM evaluation)
        relevance_score = await self._score_relevance(prompt, response)
        clarity_score = await self._score_clarity(response)
        helpfulness_score = await self._score_helpfulness(prompt, response)
        completeness_score = await self._score_completeness(prompt, response)

        overall_score = (
            relevance_score + clarity_score + helpfulness_score + completeness_score
        ) / 4

        improvement_areas = []
        if relevance_score < 0.7:
            improvement_areas.append("relevance")
        if clarity_score < 0.7:
            improvement_areas.append("clarity")
        if helpfulness_score < 0.7:
            improvement_areas.append("helpfulness")
        if completeness_score < 0.7:
            improvement_areas.append("completeness")

        quality = ConversationQuality(
            conversation_id=conversation_id,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            helpfulness_score=helpfulness_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            improvement_areas=improvement_areas,
            timestamp=datetime.now().isoformat(),
        )

        # Store quality analysis
        await self._store_quality_analysis(quality)

        # Check if improvements should be triggered
        if overall_score < self.quality_thresholds["poor"]:
            await self._trigger_quality_improvement(conversation_id, quality)

        return quality

    async def _score_relevance(self, prompt: str, response: str) -> float:
        """Score how relevant the response is to the prompt"""
        # Simple heuristic scoring (can be enhanced with embeddings/LLM)

        # Check for direct question answering
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        score = 0.5  # Base score

        # Boost score if response addresses prompt keywords
        prompt_words = set(prompt_lower.split())
        response_words = set(response_lower.split())

        # Remove common stop words
        stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "and",
            "a",
            "to",
            "are",
            "as",
            "was",
            "with",
            "for",
        }
        prompt_keywords = prompt_words - stop_words
        response_keywords = response_words - stop_words

        if prompt_keywords:
            overlap_ratio = len(prompt_keywords & response_keywords) / len(
                prompt_keywords
            )
            score += overlap_ratio * 0.3

        # Check for question patterns
        if any(
            q in prompt_lower for q in ["what", "how", "why", "when", "where", "who"]
        ):
            if len(response) > 20:  # Substantial response
                score += 0.2

        return min(1.0, score)

    async def _score_clarity(self, response: str) -> float:
        """Score how clear and understandable the response is"""
        score = 0.5  # Base score

        # Length appropriateness
        if 10 <= len(response) <= 2000:
            score += 0.2

        # Sentence structure
        sentences = response.split(".")
        if 1 <= len(sentences) <= 20:
            score += 0.2

        # Avoid excessive technical jargon (simple check)
        words = response.split()
        long_words = [w for w in words if len(w) > 12]
        if len(long_words) / max(len(words), 1) < 0.1:
            score += 0.1

        return min(1.0, score)

    async def _score_helpfulness(self, prompt: str, response: str) -> float:
        """Score how helpful the response is"""
        score = 0.5  # Base score

        # Check for actionable content
        action_words = [
            "try",
            "use",
            "consider",
            "implement",
            "apply",
            "follow",
            "check",
        ]
        if any(word in response.lower() for word in action_words):
            score += 0.2

        # Check for examples or specifics
        if any(
            marker in response.lower()
            for marker in ["example", "for instance", "such as", "like"]
        ):
            score += 0.1

        # Response length indicates effort
        if len(response) > 50:
            score += 0.2

        return min(1.0, score)

    async def _score_completeness(self, prompt: str, response: str) -> float:
        """Score how complete the response is"""
        score = 0.5  # Base score

        # Multi-part questions should have multi-part answers
        if "?" in prompt:
            question_count = prompt.count("?")
            if question_count > 1:
                # Check if response addresses multiple aspects
                if any(
                    connector in response.lower()
                    for connector in ["also", "additionally", "furthermore", "moreover"]
                ):
                    score += 0.3

        # Check for structured responses
        if any(marker in response for marker in ["1.", "2.", "-", "*", "\n"]):
            score += 0.2

        return min(1.0, score)

    async def _store_quality_analysis(self, quality: ConversationQuality):
        """Store quality analysis in database"""
        try:
            # Import evaluation engine to store the analysis
            from evaluation_engine import evaluation_engine

            if evaluation_engine:
                await evaluation_engine.log_conversation_evaluation(
                    conversation_id=quality.conversation_id,
                    relevance=quality.relevance_score,
                    clarity=quality.clarity_score,
                    completeness=quality.completeness_score,
                    accuracy=quality.helpfulness_score,  # Using helpfulness as proxy for accuracy
                    helpfulness=quality.helpfulness_score,
                    overall_score=quality.overall_score,
                    improvement_areas=quality.improvement_areas,
                )
        except Exception as e:
            logger.error(f"Failed to store quality analysis: {e}")

    async def _trigger_quality_improvement(
        self, conversation_id: str, quality: ConversationQuality
    ):
        """Trigger improvement cycle based on poor conversation quality"""
        try:
            from global_self_improvement import global_orchestrator

            if global_orchestrator:
                # Create improvement context from conversation quality
                improvement_context = {
                    "trigger": "conversation_quality",
                    "conversation_id": conversation_id,
                    "quality_score": quality.overall_score,
                    "improvement_areas": quality.improvement_areas,
                    "timestamp": quality.timestamp,
                }

                # Trigger focused improvement cycle
                await global_orchestrator.run_targeted_improvement(
                    focus_areas=["evaluation", "response_quality"],
                    context=improvement_context,
                )

                logger.info(
                    f"Triggered quality improvement for conversation {conversation_id}"
                )
        except Exception as e:
            logger.error(f"Failed to trigger quality improvement: {e}")

    async def generate_improvement_suggestions(
        self, conversation_id: str, conversation_history: List[Dict[str, str]]
    ) -> List[ConversationImprovement]:
        """Generate improvement suggestions for ongoing conversation"""
        suggestions = []

        if not conversation_history:
            return suggestions

        recent_exchanges = conversation_history[-3:]  # Last 3 exchanges

        # Analyze patterns in recent conversation
        avg_response_length = sum(
            len(ex.get("response", "")) for ex in recent_exchanges
        ) / len(recent_exchanges)

        # Suggestion 1: Response length optimization
        if avg_response_length < 50:
            suggestions.append(
                ConversationImprovement(
                    conversation_id=conversation_id,
                    improvement_type="response_length",
                    suggestion="Consider providing more detailed responses with examples and explanations",
                    confidence=0.8,
                    estimated_impact=0.6,
                    auto_apply=False,
                    context={"avg_length": avg_response_length, "threshold": 50},
                )
            )
        elif avg_response_length > 1000:
            suggestions.append(
                ConversationImprovement(
                    conversation_id=conversation_id,
                    improvement_type="response_conciseness",
                    suggestion="Consider making responses more concise and focused",
                    confidence=0.7,
                    estimated_impact=0.5,
                    auto_apply=False,
                    context={"avg_length": avg_response_length, "threshold": 1000},
                )
            )

        # Suggestion 2: Question handling
        questions_asked = sum(
            1 for ex in recent_exchanges if "?" in ex.get("prompt", "")
        )
        if questions_asked > 0:
            direct_answers = sum(
                1
                for ex in recent_exchanges
                if self._has_direct_answer(ex.get("response", ""))
            )
            if direct_answers / questions_asked < 0.7:
                suggestions.append(
                    ConversationImprovement(
                        conversation_id=conversation_id,
                        improvement_type="question_answering",
                        suggestion="Focus on providing more direct answers to questions",
                        confidence=0.9,
                        estimated_impact=0.8,
                        auto_apply=False,
                        context={
                            "questions": questions_asked,
                            "direct_answers": direct_answers,
                        },
                    )
                )

        # Store suggestions
        self.improvement_suggestions[conversation_id] = suggestions

        return suggestions

    def _has_direct_answer(self, response: str) -> bool:
        """Check if response contains direct answer patterns"""
        response_lower = response.lower()
        direct_patterns = [
            "yes",
            "no",
            "the answer is",
            "you can",
            "you should",
            "to do this",
            "the solution",
            "here's how",
            "follow these steps",
        ]
        return any(pattern in response_lower for pattern in direct_patterns)

    async def apply_conversation_feedback(
        self, conversation_id: str, feedback_type: str, feedback_data: Dict[str, Any]
    ):
        """Apply user feedback to improve future conversations"""
        try:
            # Store feedback for learning
            feedback_entry = {
                "conversation_id": conversation_id,
                "feedback_type": feedback_type,
                "feedback_data": feedback_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Trigger learning from feedback
            from knowledge_harvesting import knowledge_harvester

            if knowledge_harvester:
                await knowledge_harvester.extract_feedback_insights(feedback_entry)

            # If feedback indicates poor quality, trigger improvement
            if feedback_type in ["negative", "poor_quality", "unhelpful"]:
                await self._trigger_feedback_improvement(
                    conversation_id, feedback_entry
                )

        except Exception as e:
            logger.error(f"Failed to apply conversation feedback: {e}")

    async def _trigger_feedback_improvement(
        self, conversation_id: str, feedback: Dict[str, Any]
    ):
        """Trigger improvement cycle based on negative feedback"""
        try:
            from global_self_improvement import global_orchestrator

            if global_orchestrator:
                improvement_context = {
                    "trigger": "negative_feedback",
                    "conversation_id": conversation_id,
                    "feedback": feedback,
                    "priority": "high",
                }

                await global_orchestrator.run_targeted_improvement(
                    focus_areas=["conversation_quality", "user_satisfaction"],
                    context=improvement_context,
                )

                logger.info(
                    f"Triggered feedback-based improvement for conversation {conversation_id}"
                )
        except Exception as e:
            logger.error(f"Failed to trigger feedback improvement: {e}")

    async def get_conversation_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights and suggestions for a conversation session"""
        try:
            # Get recent conversation history
            from db import get_session_history

            history = get_session_history(session_id, limit=10)

            if not history:
                return {"insights": [], "suggestions": [], "quality_trend": "stable"}

            # Analyze conversation patterns
            conversation_data = []
            for entry in history:
                conversation_data.append(
                    {
                        "prompt": entry.prompt,
                        "response": entry.response,
                        "timestamp": (
                            entry.ts.isoformat()
                            if entry.ts
                            else datetime.now().isoformat()
                        ),
                    }
                )

            # Generate insights
            insights = await self._analyze_conversation_patterns(conversation_data)

            # Generate suggestions
            suggestions = await self.generate_improvement_suggestions(
                session_id, conversation_data
            )

            # Determine quality trend
            if len(conversation_data) >= 3:
                recent_quality = [
                    await self._quick_quality_score(ex["prompt"], ex["response"])
                    for ex in conversation_data[:3]
                ]
                older_quality = [
                    await self._quick_quality_score(ex["prompt"], ex["response"])
                    for ex in conversation_data[-3:]
                ]

                recent_avg = sum(recent_quality) / len(recent_quality)
                older_avg = sum(older_quality) / len(older_quality)

                if recent_avg > older_avg + 0.1:
                    quality_trend = "improving"
                elif recent_avg < older_avg - 0.1:
                    quality_trend = "declining"
                else:
                    quality_trend = "stable"
            else:
                quality_trend = "insufficient_data"

            return {
                "insights": insights,
                "suggestions": [asdict(s) for s in suggestions],
                "quality_trend": quality_trend,
                "conversation_count": len(conversation_data),
            }

        except Exception as e:
            logger.error(f"Failed to get conversation insights: {e}")
            return {"insights": [], "suggestions": [], "quality_trend": "error"}

    async def _analyze_conversation_patterns(
        self, conversation_data: List[Dict[str, str]]
    ) -> List[str]:
        """Analyze patterns in conversation data"""
        insights = []

        if len(conversation_data) < 2:
            return insights

        # Response time analysis (if we had timestamps)
        avg_response_length = sum(
            len(conv["response"]) for conv in conversation_data
        ) / len(conversation_data)

        if avg_response_length > 500:
            insights.append("Conversation features detailed, comprehensive responses")
        elif avg_response_length < 100:
            insights.append("Conversation features brief, concise responses")

        # Question pattern analysis
        questions = [conv for conv in conversation_data if "?" in conv["prompt"]]
        if len(questions) / len(conversation_data) > 0.7:
            insights.append("Conversation is highly question-driven")

        # Topic consistency (simple keyword analysis)
        all_words = []
        for conv in conversation_data:
            all_words.extend(conv["prompt"].lower().split())
            all_words.extend(conv["response"].lower().split())

        word_freq = {}
        for word in all_words:
            if len(word) > 4:  # Focus on meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1

        if word_freq:
            most_common = max(word_freq.items(), key=lambda x: x[1])
            if most_common[1] > 3:
                insights.append(
                    f"Conversation frequently discusses topics related to '{most_common[0]}'"
                )

        return insights

    async def _quick_quality_score(self, prompt: str, response: str) -> float:
        """Quick quality scoring for trend analysis"""
        # Simplified version of quality scoring
        relevance = await self._score_relevance(prompt, response)
        clarity = await self._score_clarity(response)
        return (relevance + clarity) / 2


# Global conversation integrator instance
conversation_integrator = ConversationIntegrator()


async def analyze_chat_quality(
    conversation_id: str, prompt: str, response: str, session_id: Optional[str] = None
) -> ConversationQuality:
    """Analyze quality of a chat interaction"""
    return await conversation_integrator.analyze_conversation_quality(
        conversation_id, prompt, response, session_id
    )


async def get_improvement_suggestions(
    conversation_id: str, conversation_history: List[Dict[str, str]]
) -> List[ConversationImprovement]:
    """Get improvement suggestions for a conversation"""
    return await conversation_integrator.generate_improvement_suggestions(
        conversation_id, conversation_history
    )


async def apply_feedback(
    conversation_id: str, feedback_type: str, feedback_data: Dict[str, Any]
):
    """Apply user feedback to the improvement system"""
    return await conversation_integrator.apply_conversation_feedback(
        conversation_id, feedback_type, feedback_data
    )


async def get_session_insights(session_id: str) -> Dict[str, Any]:
    """Get insights for a conversation session"""
    return await conversation_integrator.get_conversation_insights(session_id)
