"""
User Profiling Engine - Database Schema
========================================

Comprehensive SQLAlchemy schema for hyper-detailed user profiling with
versioning, timestamping, and append-only data integrity.
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float,
    Boolean,
    JSON,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
import json
import os
from typing import Optional, Dict, Any

Base = declarative_base()


class IdentityCore(Base):
    """Core identity information - versioned and timestamped"""

    __tablename__ = "identity_core"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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

    __tablename__ = "contact_info"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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

    __tablename__ = "medical_profile"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

    profile_type = Column(
        String(50)
    )  # condition, medication, allergy, sleep, activity, diet, test
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

    __tablename__ = "financial_status"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

    category = Column(
        String(50)
    )  # income, expense, investment, debt, balance, transaction
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

    __tablename__ = "education_profile"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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

    __tablename__ = "calendar"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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

    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now)

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
    # --- END OF ORM CLASSES ---

    """User's web presence and digital footprint"""
    __tablename__ = "web_presence"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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
    scan_date = Column(DateTime, default=func.now)

    # Privacy and sensitivity
    is_public = Column(Boolean, default=True)

    sensitivity_level = Column(String(20))  # low, medium, high


# --- MISSING ORM CLASSES ---
class InterestsHobbies(Base):
    """User interests and hobbies"""

    __tablename__ = "interests_hobbies"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

    # Core fields
    interest_type = Column(String(100))  # hobby, passion, skill, fandom, etc.
    name = Column(String(255))
    description = Column(Text)
    proficiency_level = Column(String(50))  # beginner, intermediate, expert
    enjoyment_level = Column(Float)  # 0.0 to 1.0
    frequency = Column(String(50))  # daily, weekly, monthly, etc.
    group_activity = Column(Boolean, default=False)

    # Metadata
    confidence_score = Column(Float, default=0.0)
    data_source = Column(String(100))
    is_active = Column(Boolean, default=True)


class SocialCircle(Base):
    """User's social relationships and circles"""

    __tablename__ = "social_circle"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now)
    updated_at = Column(DateTime, default=func.now, onupdate=func.now)

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


# --- UserProfileDatabase class (after all ORM classes) ---
class UserProfileDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "user_core.db")
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
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
        return {
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
            "search_history_entries": session.query(SearchHistory).count(),
            "web_presence_records": session.query(WebPresence).count(),
            "last_updated": datetime.now().isoformat(),
        }


def init_user_database(db_path: Optional[str] = None) -> UserProfileDatabase:
    """Initialize user profile database"""
    return UserProfileDatabase(db_path)


def create_tables(db_path: Optional[str] = None) -> bool:
    """Create all database tables"""
    try:
        db = UserProfileDatabase(db_path)
        db.create_tables()
        print(f"Database tables created successfully at {db.db_path}")
        return True
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return False


def migrate_database(db_path: Optional[str] = None) -> bool:
    """Perform database migrations and schema updates"""
    try:
        db = UserProfileDatabase(db_path)
        session = db.get_session()

        # Check for missing columns and add them
        migrations_applied = []

        try:
            # Check if feedback columns exist in Calendar table
            session.execute("SELECT stress_level FROM calendar LIMIT 1")
        except Exception:
            # Add stress_level column if missing
            try:
                session.execute("ALTER TABLE calendar ADD COLUMN stress_level REAL")
                migrations_applied.append("Added stress_level column to calendar table")
            except Exception as e:
                print(f"Could not add stress_level column: {e}")

        try:
            # Check if external_id exists in Calendar table
            session.execute("SELECT external_id FROM calendar LIMIT 1")
        except Exception:
            # Add external_id column if missing
            try:
                session.execute("ALTER TABLE calendar ADD COLUMN external_id TEXT")
                migrations_applied.append("Added external_id column to calendar table")
            except Exception as e:
                print(f"Could not add external_id column: {e}")

        try:
            # Check if recurrence_pattern exists in Calendar table
            session.execute("SELECT recurrence_pattern FROM calendar LIMIT 1")
        except Exception:
            # Add recurrence_pattern column if missing
            try:
                session.execute(
                    "ALTER TABLE calendar ADD COLUMN recurrence_pattern TEXT"
                )
                migrations_applied.append(
                    "Added recurrence_pattern column to calendar table"
                )
            except Exception as e:
                print(f"Could not add recurrence_pattern column: {e}")

        # Add indexes for better performance
        try:
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_calendar_start_time ON calendar(start_time)"
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_calendar_is_active ON calendar(is_active)"
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_financial_created_at ON financial_status(created_at)"
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at)"
            )
            migrations_applied.append("Added performance indexes")
        except Exception as e:
            print(f"Could not add indexes: {e}")

        session.commit()

        if migrations_applied:
            print("Database migrations applied:")
            for migration in migrations_applied:
                print(f"  - {migration}")
        else:
            print("No migrations needed - database is up to date")

        return True

    except Exception as e:
        print(f"Error during database migration: {e}")
        return False
    finally:
        session.close()


def vacuum_database(db_path: Optional[str] = None) -> bool:
    """Vacuum database to reclaim space and optimize performance"""
    try:
        db = UserProfileDatabase(db_path)
        session = db.get_session()

        # Run VACUUM to reclaim space
        session.execute("VACUUM")

        # Update statistics
        session.execute("ANALYZE")

        print(f"Database optimized successfully at {db.db_path}")
        return True

    except Exception as e:
        print(f"Error optimizing database: {e}")
        return False
    finally:
        session.close()


def backup_database(
    db_path: Optional[str] = None, backup_path: Optional[str] = None
) -> bool:
    """Create a backup of the database"""
    try:
        import shutil
        from datetime import datetime

        db = UserProfileDatabase(db_path)

        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db.db_path}.backup.{timestamp}"

        shutil.copy2(db.db_path, backup_path)
        print(f"Database backed up to {backup_path}")
        return True

    except Exception as e:
        print(f"Error backing up database: {e}")
        return False


def get_database_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get database statistics and health information"""
    try:
        db = UserProfileDatabase(db_path)
        session = db.get_session()

        stats = {
            "database_path": db.db_path,
            "table_counts": {},
            "database_size_mb": 0,
            "last_updated": datetime.now().isoformat(),
        }

        # Get table counts
        table_names = [
            "identity_core",
            "contact_info",
            "medical_profile",
            "financial_status",
            "education_profile",
            "calendar",
            "search_history",
            "web_presence",
            "interests_hobbies",
            "social_circle",
        ]

        for table_name in table_names:
            try:
                result = session.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = result.scalar()
                stats["table_counts"][table_name] = count
            except Exception as e:
                stats["table_counts"][table_name] = f"Error: {e}"

        # Get database file size
        try:
            import os

            if os.path.exists(db.db_path):
                size_bytes = os.path.getsize(db.db_path)
                stats["database_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        except Exception:
            stats["database_size_mb"] = "Unknown"

        return stats

    except Exception as e:
        return {"error": f"Could not get database stats: {e}"}
    finally:
        session.close()


def cleanup_old_data(db_path: Optional[str] = None, days_to_keep: int = 365) -> bool:
    """Clean up old data based on retention policy"""
    try:
        from datetime import datetime, timedelta

        db = UserProfileDatabase(db_path)
        session = db.get_session()

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Clean up old search history
        old_searches = (
            session.query(SearchHistory)
            .filter(SearchHistory.created_at < cutoff_date)
            .delete()
        )

        # Clean up old financial records (keep recent ones)
        old_financial = (
            session.query(FinancialStatus)
            .filter(FinancialStatus.created_at < cutoff_date)
            .delete()
        )

        session.commit()

        print(
            f"Cleaned up {old_searches} old search records and {old_financial} old financial records"
        )
        return True

    except Exception as e:
        print(f"Error cleaning up old data: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    """Database management CLI"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python user_profile_schema.py <command> [db_path]")
        print("Commands:")
        print("  create     - Create database tables")
        print("  migrate    - Run database migrations")
        print("  vacuum     - Optimize database")
        print("  backup     - Create database backup")
        print("  stats      - Show database statistics")
        print("  cleanup    - Clean up old data (365 days retention)")
        sys.exit(1)

    command = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "create":
        create_tables(db_path)
    elif command == "migrate":
        migrate_database(db_path)
    elif command == "vacuum":
        vacuum_database(db_path)
    elif command == "backup":
        backup_database(db_path)
    elif command == "stats":
        stats = get_database_stats(db_path)
        print("Database Statistics:")
        for key, value in stats.items():
            if key == "table_counts":
                print(f"  {key}:")
                for table, count in value.items():
                    print(f"    {table}: {count}")
            else:
                print(f"  {key}: {value}")
    elif command == "cleanup":
        cleanup_old_data(db_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
