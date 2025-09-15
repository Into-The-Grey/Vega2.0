"""
User Profiling Engine - Database Schema
========================================

Comprehensive SQLAlchemy schema for hyper-detailed user profiling with 
versioning, timestamping, and append-only data integrity.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any

Base = declarative_base()

class IdentityCore(Base):
    """Core identity information - versioned and timestamped"""
    __tablename__ = 'identity_core'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    full_name = Column(String(255))
    aliases = Column(JSON)  # List of nicknames/aliases
    birth_date = Column(DateTime)
    gender = Column(String(50))
    sexual_identity = Column(String(100))
    citizenship = Column(String(100))
    primary_language = Column(String(50))
    
    # Metadata
    confidence_score = Column(Float, default=0.0)  # How confident we are in this data
    data_source = Column(String(100))  # Where this info came from
    is_active = Column(Boolean, default=True)

class ContactInfo(Base):
    """Contact information with versioning"""
    __tablename__ = 'contact_info'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    contact_type = Column(String(50))  # phone, email, username, social_handle
    contact_value = Column(String(255))
    platform = Column(String(100))  # Twitter, Reddit, GitHub, etc.
    is_primary = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class MedicalProfile(Base):
    """Medical and health information"""
    __tablename__ = 'medical_profile'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    profile_type = Column(String(50))  # condition, medication, allergy, sleep, activity, diet, test
    title = Column(String(255))
    description = Column(Text)
    severity = Column(String(50))  # low, medium, high, critical
    status = Column(String(50))  # active, inactive, resolved, chronic
    
    # JSON fields for structured data
    details = Column(JSON)  # Flexible storage for specific data types
    metrics = Column(JSON)  # Numerical data, trends, etc.
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class FinancialStatus(Base):
    """Financial information with privacy-conscious abstractions"""
    __tablename__ = 'financial_status'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    category = Column(String(50))  # income, expense, investment, debt, balance, transaction
    subcategory = Column(String(100))  # salary, rent, groceries, etc.
    
    # Abstracted values (not exact amounts)
    amount_tier = Column(String(20))  # very_low, low, medium, high, very_high
    frequency = Column(String(20))  # daily, weekly, monthly, yearly, one_time
    
    # Trends and patterns
    trend = Column(String(20))  # increasing, decreasing, stable, volatile
    seasonality = Column(JSON)  # Seasonal patterns
    
    # Behavioral insights
    spending_pattern = Column(String(100))
    financial_stress_indicator = Column(Float)  # 0.0 to 1.0
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class EducationProfile(Base):
    """Educational background and current studies"""
    __tablename__ = 'education_profile'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    education_type = Column(String(50))  # institution, degree, course, assignment, goal
    institution_name = Column(String(255))
    program_name = Column(String(255))
    
    # Academic details
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    grade_gpa = Column(Float)
    status = Column(String(50))  # current, completed, dropped, planned
    
    # Course-specific data
    course_details = Column(JSON)  # Syllabus, schedule, assignments
    performance_metrics = Column(JSON)  # Grades, attendance, etc.
    
    # Career alignment
    career_relevance = Column(Float)  # How relevant to career goals
    skill_development = Column(JSON)  # Skills being developed
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class Calendar(Base):
    """Calendar events and scheduling"""
    __tablename__ = 'calendar'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Event details
    title = Column(String(255))
    description = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    location = Column(String(255))
    
    # Event classification
    event_type = Column(String(50))  # work, school, personal, medical, social
    priority = Column(String(20))  # low, medium, high, critical
    stress_level = Column(Float)  # Predicted stress impact
    
    # Recurrence
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(JSON)
    
    # Source and sync
    source = Column(String(50))  # google, apple, manual
    external_id = Column(String(255))  # ID from external calendar
    
    # Status tracking
    status = Column(String(20))  # scheduled, completed, missed, cancelled
    actual_duration = Column(Integer)  # Minutes
    
    # Metadata
    is_active = Column(Boolean, default=True)

class InterestsHobbies(Base):
    """User interests, hobbies, and preferences"""
    __tablename__ = 'interests_hobbies'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    interest_type = Column(String(50))  # current, historical, wishlist
    category = Column(String(100))  # technology, music, sports, etc.
    title = Column(String(255))
    description = Column(Text)
    
    # Engagement metrics
    engagement_level = Column(Float)  # 0.0 to 1.0
    time_investment = Column(String(50))  # hours per week
    skill_level = Column(String(50))  # beginner, intermediate, advanced, expert
    
    # Ranking and priority
    importance_rank = Column(Integer)  # User's priority ranking
    trend = Column(String(20))  # growing, stable, declining
    
    # Social aspects
    is_social_activity = Column(Boolean, default=False)
    community_involvement = Column(JSON)  # Online/offline communities
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class SocialCircle(Base):
    """Social relationships and connections"""
    __tablename__ = 'social_circle'
    
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Identity
    name = Column(String(255))
    aliases = Column(JSON)  # Nicknames, handles
    
    # Relationship details
    relationship_type = Column(String(100))  # friend, family, romantic, colleague, etc.
    relationship_strength = Column(Float)  # 0.0 to 1.0
    trust_level = Column(String(20))  # low, medium, high
    
    # Personal details
    birthday = Column(DateTime)
    contact_info = Column(JSON)
    
    # Interaction guidelines
    topics_to_avoid = Column(JSON)  # List of sensitive topics
    preferred_communication = Column(String(100))  # text, call, email, etc.
    communication_frequency = Column(String(50))  # daily, weekly, monthly, etc.
    
    # Contextual intelligence
    personality_notes = Column(Text)
    shared_interests = Column(JSON)
    important_dates = Column(JSON)  # Anniversaries, achievements, etc.
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)

class SearchHistory(Base):
    """User search and query history"""
    __tablename__ = 'search_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Query details
    keyword = Column(String(500))
    query_type = Column(String(50))  # web, ai_prompt, cli, api
    source = Column(String(100))  # google, vega, chatgpt, etc.
    
    # Context
    session_id = Column(String(255))
    user_intent = Column(String(255))  # Inferred intent
    
    # Results and engagement
    results_found = Column(Integer)
    result_clicked = Column(Boolean, default=False)
    satisfaction_score = Column(Float)  # If available
    
    # Categorization
    topic_category = Column(String(100))
    urgency_level = Column(String(20))  # low, medium, high
    
    # Privacy
    is_sensitive = Column(Boolean, default=False)
    anonymized = Column(Boolean, default=False)

class WebPresence(Base):
    """User's web presence and digital footprint"""
    __tablename__ = 'web_presence'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Source details
    platform = Column(String(100))  # reddit, github, twitter, etc.
    url = Column(String(500))
    username = Column(String(255))
    
    # Content analysis
    content_type = Column(String(50))  # profile, post, comment, bio
    content_summary = Column(Text)
    sentiment = Column(String(20))  # positive, negative, neutral
    
    # Metadata extraction
    extracted_interests = Column(JSON)
    extracted_skills = Column(JSON)
    extracted_locations = Column(JSON)
    extracted_connections = Column(JSON)
    
    # Verification and confidence
    is_verified = Column(Boolean, default=False)
    confidence_score = Column(Float, default=0.0)
    scan_date = Column(DateTime, default=func.now())
    
    # Privacy and sensitivity
    is_public = Column(Boolean, default=True)
    sensitivity_level = Column(String(20))  # low, medium, high

class UserProfileDatabase:
    """Main database management class for user profiling"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'user_core.db')
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def backup_database(self, backup_path: str):
        """Create backup of database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    def get_user_summary(self) -> Dict[str, Any]:
        """Get high-level summary of user profile"""
        session = self.get_session()
        try:
            summary = {
                'identity': session.query(IdentityCore).filter(IdentityCore.is_active == True).count(),
                'contacts': session.query(ContactInfo).filter(ContactInfo.is_active == True).count(),
                'medical_records': session.query(MedicalProfile).filter(MedicalProfile.is_active == True).count(),
                'financial_records': session.query(FinancialStatus).filter(FinancialStatus.is_active == True).count(),
                'education_records': session.query(EducationProfile).filter(EducationProfile.is_active == True).count(),
                'calendar_events': session.query(Calendar).filter(Calendar.is_active == True).count(),
                'interests': session.query(InterestsHobbies).filter(InterestsHobbies.is_active == True).count(),
                'social_connections': session.query(SocialCircle).filter(SocialCircle.is_active == True).count(),
                'search_history_entries': session.query(SearchHistory).count(),
                'web_presence_records': session.query(WebPresence).count(),
                'last_updated': datetime.now().isoformat()
            }
            return summary
        finally:
            session.close()

# Initialize database instance
def init_user_database(db_path: str = None) -> UserProfileDatabase:
    """Initialize user profile database"""
    return UserProfileDatabase(db_path)

if __name__ == "__main__":
    # Test database creation
    db = init_user_database()
    print("User profile database schema created successfully!")
    print(f"Database location: {db.db_path}")
    
    # Print summary
    summary = db.get_user_summary()
    print("\nDatabase Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")        shutil.copy2(self.db_path, backup_path)

    def get_user_summary(self) -> Dict[str, Any]:
        """Get high-level summary of user profile"""
        session = self.get_session()
        try:
            summary = {
                "identity": session.query(IdentityCore)
                .filter(IdentityCore.is_active == True)
                .count(),
                "contacts": session.query(ContactInfo)
                .filter(ContactInfo.is_active == True)
                .count(),
                "medical_records": session.query(MedicalProfile)
                .filter(MedicalProfile.is_active == True)
                .count(),
                "financial_records": session.query(FinancialStatus)
                .filter(FinancialStatus.is_active == True)
                .count(),
                "education_records": session.query(EducationProfile)
                .filter(EducationProfile.is_active == True)
                .count(),
                "calendar_events": session.query(Calendar)
                .filter(Calendar.is_active == True)
                .count(),
                "interests": session.query(InterestsHobbies)
                .filter(InterestsHobbies.is_active == True)
                .count(),
                "social_connections": session.query(SocialCircle)
                .filter(SocialCircle.is_active == True)
                .count(),
                "search_history_entries": session.query(SearchHistory).count(),
                "web_presence_records": session.query(WebPresence).count(),
                "last_updated": datetime.now().isoformat(),
            }
            return summary
        finally:
            session.close()


# Initialize database instance
def init_user_database(db_path: str = None) -> UserProfileDatabase:
    """Initialize user profile database"""
    return UserProfileDatabase(db_path)


if __name__ == "__main__":
    # Test database creation
    db = init_user_database()
    print("User profile database schema created successfully!")
    print(f"Database location: {db.db_path}")

    # Print summary
    summary = db.get_user_summary()
    print("\nDatabase Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
