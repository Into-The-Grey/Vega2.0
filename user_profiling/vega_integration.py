"""
Vega2.0 User Profiling Integration
==================================

Integrates the User Profiling Engine (UPE) with the main Vega2.0 FastAPI system.
Provides contextual intelligence injection into chat responses and profile management APIs.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from user_profiling.database.user_profile_schema import UserProfileDatabase
from user_profiling.user_profile_daemon import (
    UserProfileDaemon, DaemonConfig, DailyBriefingGenerator, 
    UnderstandingScoreCalculator
)
from user_profiling.engines.persona_engine import PersonaEngine, PersonaMode
from user_profiling.collectors.profile_intel_collector import ProfileIntelCollector

logger = logging.getLogger(__name__)

# Global daemon instance
user_profile_daemon: Optional[UserProfileDaemon] = None

class UserProfileConfig(BaseModel):
    """Configuration for user profiling integration"""
    enable_profiling: bool = Field(default=True, description="Enable user profiling")
    enable_contextual_responses: bool = Field(default=True, description="Enable contextual chat responses")
    enable_auto_enrichment: bool = Field(default=True, description="Enable autonomous enrichment")
    daemon_scan_interval_hours: int = Field(default=24, description="Full scan interval hours")
    persona_update_interval_minutes: int = Field(default=30, description="Persona update interval")
    max_context_injection_length: int = Field(default=500, description="Max context injection length")
    privacy_mode: str = Field(default="balanced", description="Privacy mode: strict, balanced, open")
    
class ContextualChatRequest(BaseModel):
    """Request model for contextual chat"""
    prompt: str
    session_id: Optional[str] = None
    include_persona: bool = True
    include_context: bool = True
    context_categories: List[str] = Field(default_factory=lambda: ["calendar", "academic", "personal"])

class ContextualChatResponse(BaseModel):
    """Response model for contextual chat"""
    response: str
    persona_mode: str
    context_applied: Dict[str, Any]
    understanding_score: float
    suggestions: List[str] = Field(default_factory=list)

class ProfileSummaryResponse(BaseModel):
    """User profile summary response"""
    identity: Dict[str, Any]
    current_persona: str
    understanding_score: float
    recent_activity: Dict[str, Any]
    upcoming_events: List[Dict[str, Any]]
    recommendations: List[str]
    last_updated: str

class ContextualIntelligenceEngine:
    """Contextual intelligence engine for chat enhancement"""
    
    def __init__(self, db: UserProfileDatabase, persona_engine: PersonaEngine):
        self.db = db
        self.persona_engine = persona_engine
        self.understanding_calculator = UnderstandingScoreCalculator(db)
        
    async def enhance_chat_response(
        self, 
        original_response: str, 
        user_prompt: str,
        session_id: str = None,
        config: UserProfileConfig = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Enhance chat response with contextual intelligence"""
        
        config = config or UserProfileConfig()
        context_applied = {}
        
        if not config.enable_contextual_responses:
            return original_response, context_applied
        
        try:
            # Get current persona context
            persona_summary = await self.persona_engine.get_persona_summary()
            current_mode = persona_summary.get('current_mode', PersonaMode.DEFAULT)
            
            # Get contextual information
            context_info = await self._gather_context_information(user_prompt, session_id)
            
            # Apply persona-specific response modification
            enhanced_response = await self._apply_persona_enhancement(
                original_response, user_prompt, current_mode, context_info
            )
            
            # Add contextual awareness
            if config.include_context and context_info:
                enhanced_response = await self._inject_contextual_awareness(
                    enhanced_response, context_info, config.max_context_injection_length
                )
            
            context_applied = {
                'persona_mode': current_mode.value,
                'context_categories': list(context_info.keys()),
                'enhancement_applied': True,
                'context_summary': self._summarize_context(context_info)
            }
        
        except Exception as e:
            logger.error(f"Error enhancing chat response: {e}")
            enhanced_response = original_response
            context_applied = {'error': str(e), 'enhancement_applied': False}
        
        return enhanced_response, context_applied
    
    async def _gather_context_information(self, prompt: str, session_id: str = None) -> Dict[str, Any]:
        """Gather relevant contextual information"""
        context = {}
        session = self.db.get_session()
        
        try:
            # Calendar context
            context['calendar'] = await self._get_calendar_context(session)
            
            # Academic context
            context['academic'] = await self._get_academic_context(session)
            
            # Personal context
            context['personal'] = await self._get_personal_context(session)
            
            # Social context
            context['social'] = await self._get_social_context(session)
            
            # Recent interaction context
            if session_id:
                context['recent_interactions'] = await self._get_recent_interactions(session_id)
        
        finally:
            session.close()
        
        return context
    
    async def _get_calendar_context(self, session) -> Dict[str, Any]:
        """Get calendar context for today and near future"""
        from user_profiling.database.user_profile_schema import Calendar
        
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = today_start + timedelta(days=7)
        
        # Today's events
        today_events = session.query(Calendar).filter(
            Calendar.start_time >= today_start,
            Calendar.start_time < today_start + timedelta(days=1),
            Calendar.is_active == True
        ).order_by(Calendar.start_time).all()
        
        # Upcoming important events
        upcoming_events = session.query(Calendar).filter(
            Calendar.start_time >= today_start + timedelta(days=1),
            Calendar.start_time < week_end,
            Calendar.importance_score > 0.7,
            Calendar.is_active == True
        ).order_by(Calendar.start_time).limit(3).all()
        
        # Calculate current stress level
        current_hour = now.hour
        current_events = [e for e in today_events if e.start_time.hour <= current_hour <= (e.end_time.hour if e.end_time else e.start_time.hour + 1)]
        stress_level = max([e.stress_level for e in current_events if e.stress_level], default=0.3)
        
        return {
            'today_events_count': len(today_events),
            'current_stress_level': stress_level,
            'next_event': {
                'title': today_events[0].title,
                'time': today_events[0].start_time.strftime('%H:%M'),
                'type': today_events[0].event_type
            } if today_events else None,
            'upcoming_important': [
                {
                    'title': e.title,
                    'date': e.start_time.strftime('%m/%d'),
                    'importance': e.importance_score
                }
                for e in upcoming_events
            ]
        }
    
    async def _get_academic_context(self, session) -> Dict[str, Any]:
        """Get academic context"""
        from user_profiling.database.user_profile_schema import EducationProfile
        
        # Active courses
        courses = session.query(EducationProfile).filter(
            EducationProfile.education_type == 'course',
            EducationProfile.status == 'current',
            EducationProfile.is_active == True
        ).all()
        
        # Find urgent deadlines
        urgent_deadlines = []
        for course in courses:
            course_details = course.course_details or {}
            assignments = course_details.get('assignments', [])
            
            for assignment in assignments:
                if assignment.get('due_date'):
                    try:
                        due_date = datetime.fromisoformat(assignment['due_date'])
                        days_until = (due_date - datetime.now()).days
                        if 0 <= days_until <= 3:  # Next 3 days
                            urgent_deadlines.append({
                                'title': assignment['title'],
                                'course': course.program_name,
                                'days_until': days_until
                            })
                    except ValueError:
                        continue
        
        return {
            'active_courses': len(courses),
            'course_names': [c.program_name for c in courses],
            'urgent_deadlines': urgent_deadlines,
            'academic_stress': len(urgent_deadlines) / 3.0  # Normalize
        }
    
    async def _get_personal_context(self, session) -> Dict[str, Any]:
        """Get personal context"""
        from user_profiling.database.user_profile_schema import InterestsHobbies
        
        # Top interests
        interests = session.query(InterestsHobbies).filter(
            InterestsHobbies.interest_type == 'current',
            InterestsHobbies.is_active == True
        ).order_by(InterestsHobbies.engagement_level.desc()).limit(3).all()
        
        return {
            'top_interests': [i.title for i in interests],
            'mood_indicators': {
                'interest_engagement': sum(i.engagement_level for i in interests) / len(interests) if interests else 0.5
            }
        }
    
    async def _get_social_context(self, session) -> Dict[str, Any]:
        """Get social context"""
        from user_profiling.database.user_profile_schema import SocialCircle
        
        # Recent social connections
        connections = session.query(SocialCircle).filter(
            SocialCircle.is_active == True
        ).all()
        
        return {
            'total_connections': len(connections),
            'relationship_types': {
                'close_friends': len([c for c in connections if c.relationship_type == 'friend' and c.closeness_level > 0.7]),
                'family': len([c for c in connections if c.relationship_type == 'family']),
                'professional': len([c for c in connections if c.relationship_type == 'colleague'])
            }
        }
    
    async def _get_recent_interactions(self, session_id: str) -> Dict[str, Any]:
        """Get recent interaction context (would integrate with main chat history)"""
        # This would typically query the main conversation database
        # For now, return placeholder
        return {
            'session_id': session_id,
            'recent_topics': [],  # Would be extracted from recent messages
            'conversation_style': 'balanced'
        }
    
    async def _apply_persona_enhancement(
        self, 
        response: str, 
        prompt: str, 
        persona_mode: PersonaMode, 
        context: Dict[str, Any]
    ) -> str:
        """Apply persona-specific response enhancement"""
        
        # Get persona behavior settings
        behavior = await self.persona_engine.get_current_behavior_settings()
        
        # Apply persona-specific modifications
        if persona_mode == PersonaMode.ACADEMIC:
            if context.get('academic', {}).get('urgent_deadlines'):
                response = self._add_academic_urgency_awareness(response, context['academic'])
        
        elif persona_mode == PersonaMode.FOCUSED:
            # More concise responses in focused mode
            response = self._make_response_more_concise(response)
        
        elif persona_mode == PersonaMode.BURNOUT:
            # More supportive and less demanding tone
            response = self._add_supportive_tone(response)
        
        elif persona_mode == PersonaMode.SOCIAL:
            # More engaging and conversational
            response = self._add_social_engagement(response)
        
        return response
    
    def _add_academic_urgency_awareness(self, response: str, academic_context: Dict[str, Any]) -> str:
        """Add academic urgency awareness to response"""
        urgent_deadlines = academic_context.get('urgent_deadlines', [])
        
        if urgent_deadlines:
            urgency_note = f"\n\n*Note: I see you have {len(urgent_deadlines)} upcoming deadline(s). Let me know if you need help prioritizing or managing your academic tasks.*"
            response += urgency_note
        
        return response
    
    def _make_response_more_concise(self, response: str) -> str:
        """Make response more concise for focused mode"""
        # Simplified implementation - could use more sophisticated text processing
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def _add_supportive_tone(self, response: str) -> str:
        """Add supportive tone for burnout mode"""
        supportive_phrases = [
            "Take your time with this.",
            "Remember to take breaks when needed.",
            "You're doing great, even when it doesn't feel like it."
        ]
        
        # Add a random supportive phrase (simplified implementation)
        import random
        if random.random() < 0.3:  # 30% chance
            response += f"\n\n{random.choice(supportive_phrases)}"
        
        return response
    
    def _add_social_engagement(self, response: str) -> str:
        """Add social engagement for social mode"""
        if not response.endswith('?') and len(response.split()) > 10:
            response += " What do you think about that?"
        
        return response
    
    async def _inject_contextual_awareness(
        self, 
        response: str, 
        context: Dict[str, Any], 
        max_length: int
    ) -> str:
        """Inject contextual awareness into response"""
        
        contextual_insights = []
        
        # Calendar insights
        calendar_context = context.get('calendar', {})
        if calendar_context.get('current_stress_level', 0) > 0.7:
            contextual_insights.append("I notice you might be in a high-stress period based on your schedule.")
        
        if calendar_context.get('next_event'):
            next_event = calendar_context['next_event']
            contextual_insights.append(f"Your next event is {next_event['title']} at {next_event['time']}.")
        
        # Academic insights
        academic_context = context.get('academic', {})
        urgent_deadlines = academic_context.get('urgent_deadlines', [])
        if urgent_deadlines:
            deadline_text = f"upcoming deadline{'s' if len(urgent_deadlines) > 1 else ''}"
            contextual_insights.append(f"I see you have {len(urgent_deadlines)} {deadline_text} this week.")
        
        # Combine insights
        if contextual_insights:
            context_text = " ".join(contextual_insights)
            if len(context_text) <= max_length:
                response = f"{response}\n\n*Context: {context_text}*"
        
        return response
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Summarize context for response metadata"""
        summary = {}
        
        if 'calendar' in context:
            cal = context['calendar']
            summary['calendar'] = f"{cal.get('today_events_count', 0)} events today, stress level: {cal.get('current_stress_level', 0):.1f}"
        
        if 'academic' in context:
            acad = context['academic']
            summary['academic'] = f"{acad.get('active_courses', 0)} courses, {len(acad.get('urgent_deadlines', []))} urgent deadlines"
        
        return summary

class UserProfileManager:
    """User profile management and API endpoints"""
    
    def __init__(self, db: UserProfileDatabase, daemon: UserProfileDaemon):
        self.db = db
        self.daemon = daemon
        self.briefing_generator = DailyBriefingGenerator(db)
        self.understanding_calculator = UnderstandingScoreCalculator(db)
    
    async def get_profile_summary(self) -> ProfileSummaryResponse:
        """Get comprehensive profile summary"""
        session = self.db.get_session()
        
        try:
            # Get identity info
            from user_profiling.database.user_profile_schema import IdentityCore, Calendar
            
            identity_record = session.query(IdentityCore).filter(
                IdentityCore.is_active == True
            ).first()
            
            identity = {
                'name': identity_record.full_name if identity_record else 'Unknown',
                'age': identity_record.age if identity_record else None,
                'location': identity_record.primary_location if identity_record else None
            }
            
            # Get current persona
            persona_summary = await self.daemon.persona_engine.get_persona_summary()
            current_persona = persona_summary.get('current_mode', PersonaMode.DEFAULT).value
            
            # Get understanding score
            understanding_score = self.understanding_calculator.calculate_understanding_score()
            
            # Get upcoming events
            now = datetime.now()
            upcoming_events = session.query(Calendar).filter(
                Calendar.start_time >= now,
                Calendar.start_time <= now + timedelta(days=7),
                Calendar.is_active == True
            ).order_by(Calendar.start_time).limit(5).all()
            
            upcoming = [
                {
                    'title': e.title,
                    'date': e.start_time.isoformat(),
                    'type': e.event_type,
                    'importance': e.importance_score
                }
                for e in upcoming_events
            ]
            
            # Generate recommendations
            recommendations = await self._generate_current_recommendations()
            
            return ProfileSummaryResponse(
                identity=identity,
                current_persona=current_persona,
                understanding_score=understanding_score.overall_score,
                recent_activity=persona_summary.get('recent_activity', {}),
                upcoming_events=upcoming,
                recommendations=recommendations,
                last_updated=datetime.now().isoformat()
            )
        
        finally:
            session.close()
    
    async def _generate_current_recommendations(self) -> List[str]:
        """Generate current recommendations for the user"""
        recommendations = []
        
        # Get today's briefing
        try:
            briefing = await self.briefing_generator.generate_daily_briefing()
            briefing_recommendations = briefing.get('sections', {}).get('recommendations', [])
            recommendations.extend(briefing_recommendations)
        except Exception as e:
            logger.error(f"Error getting briefing recommendations: {e}")
        
        # Add default recommendations if none found
        if not recommendations:
            recommendations = [
                "Stay hydrated and take regular breaks",
                "Review your calendar for upcoming events",
                "Consider updating your profile information for better personalization"
            ]
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def trigger_profile_scan(self, scan_type: str = "mini") -> Dict[str, Any]:
        """Trigger manual profile scan"""
        try:
            if scan_type == "full":
                await self.daemon._run_full_scan()
            else:
                await self.daemon._run_mini_scan()
            
            return {
                'status': 'success',
                'scan_type': scan_type,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_daily_briefing(self, date: str = None) -> Dict[str, Any]:
        """Get daily briefing for specified date"""
        try:
            target_date = datetime.fromisoformat(date) if date else datetime.now()
            briefing = await self.briefing_generator.generate_daily_briefing(target_date)
            return briefing
        
        except Exception as e:
            logger.error(f"Error getting daily briefing: {e}")
            return {'error': str(e)}
    
    async def update_profile_settings(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """Update profile settings"""
        try:
            # Update daemon configuration
            for key, value in settings.items():
                if hasattr(self.daemon.config, key):
                    setattr(self.daemon.config, key, value)
            
            return {'status': 'success', 'updated_settings': list(settings.keys())}
        
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# FastAPI Integration Functions
async def startup_user_profiling():
    """Initialize user profiling on app startup"""
    global user_profile_daemon
    
    try:
        # Initialize database
        db = UserProfileDatabase()
        db.create_tables()
        
        # Initialize daemon with configuration
        config = DaemonConfig(
            scan_interval_hours=24,
            enable_intelligence_collection=True,
            enable_calendar_sync=True,
            enable_financial_monitoring=True,
            enable_educational_analysis=True,
            enable_persona_tracking=True
        )
        
        user_profile_daemon = UserProfileDaemon(config)
        
        # Start daemon in background
        asyncio.create_task(user_profile_daemon.start_daemon())
        
        logger.info("User profiling system initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize user profiling: {e}")

async def shutdown_user_profiling():
    """Shutdown user profiling on app shutdown"""
    global user_profile_daemon
    
    if user_profile_daemon:
        user_profile_daemon.running = False
        logger.info("User profiling system shutdown")

def get_contextual_intelligence() -> ContextualIntelligenceEngine:
    """Dependency to get contextual intelligence engine"""
    if not user_profile_daemon:
        raise HTTPException(status_code=503, detail="User profiling system not available")
    
    return ContextualIntelligenceEngine(
        user_profile_daemon.db, 
        user_profile_daemon.persona_engine
    )

def get_profile_manager() -> UserProfileManager:
    """Dependency to get profile manager"""
    if not user_profile_daemon:
        raise HTTPException(status_code=503, detail="User profiling system not available")
    
    return UserProfileManager(user_profile_daemon.db, user_profile_daemon)

# Enhanced chat function for main app integration
async def enhanced_chat_response(
    original_response: str,
    user_prompt: str,
    session_id: str = None,
    intelligence_engine: ContextualIntelligenceEngine = Depends(get_contextual_intelligence)
) -> ContextualChatResponse:
    """Enhanced chat response with contextual intelligence"""
    
    try:
        enhanced_response, context_applied = await intelligence_engine.enhance_chat_response(
            original_response, user_prompt, session_id
        )
        
        # Get understanding score
        understanding_score = intelligence_engine.understanding_calculator.calculate_understanding_score()
        
        # Generate suggestions
        suggestions = await _generate_response_suggestions(context_applied, user_prompt)
        
        return ContextualChatResponse(
            response=enhanced_response,
            persona_mode=context_applied.get('persona_mode', 'default'),
            context_applied=context_applied,
            understanding_score=understanding_score.overall_score,
            suggestions=suggestions
        )
    
    except Exception as e:
        logger.error(f"Error in enhanced chat response: {e}")
        return ContextualChatResponse(
            response=original_response,
            persona_mode='default',
            context_applied={'error': str(e)},
            understanding_score=0.5,
            suggestions=[]
        )

async def _generate_response_suggestions(context: Dict[str, Any], prompt: str) -> List[str]:
    """Generate follow-up suggestions based on context"""
    suggestions = []
    
    # Calendar-based suggestions
    if 'calendar' in context.get('context_categories', []):
        suggestions.append("Would you like me to help you prepare for your upcoming events?")
    
    # Academic-based suggestions
    if 'academic' in context.get('context_categories', []):
        suggestions.append("Need help with study planning or deadline management?")
    
    # General suggestions
    suggestions.extend([
        "What would you like to explore next?",
        "Is there anything specific you'd like me to help you with?"
    ])
    
    return suggestions[:3]  # Limit to 3 suggestions