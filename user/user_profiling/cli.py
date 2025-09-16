#!/usr/bin/env python3
"""
User Profiling CLI Commands
===========================

Command-line interface for the User Profiling Engine (UPE).
Provides easy access to profiling functions and system management.
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
import typer
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from user.user_profiling.database.user_profile_schema import UserProfileDatabase
from user.user_profiling.user_profile_daemon import UserProfileDaemon, DaemonConfig
from user.user_profiling.engines.persona_engine import PersonaEngine
from user.user_profiling.vega_integration import UserProfileManager, DailyBriefingGenerator

app = typer.Typer(help="User Profiling Engine CLI")


@app.command()
def init_database(
    db_path: Optional[str] = typer.Option(
        None, help="Database path (default: user_profiling.db)"
    )
):
    """Initialize user profiling database"""
    typer.echo("🗄️  Initializing user profiling database...")

    try:
        db = UserProfileDatabase(db_path)
        db.create_tables()
        typer.echo(f"✅ Database initialized at: {db.db_path}")
    except Exception as e:
        typer.echo(f"❌ Database initialization failed: {e}")
        raise typer.Exit(1)


@app.command()
def status(db_path: Optional[str] = typer.Option(None, help="Database path")):
    """Check user profiling system status"""
    typer.echo("🔍 Checking user profiling system status...")

    try:
        db = UserProfileDatabase(db_path)
        session = db.get_session()

        # Check table counts
    from user.user_profiling.database.user_profile_schema import (
            IdentityCore,
            ContactInfo,
            Calendar,
            EducationProfile,
            FinancialStatus,
            SocialCircle,
            InterestsHobbies,
        )

        counts = {
            "Identity Records": session.query(IdentityCore).count(),
            "Contact Records": session.query(ContactInfo).count(),
            "Calendar Events": session.query(Calendar).count(),
            "Education Records": session.query(EducationProfile).count(),
            "Financial Records": session.query(FinancialStatus).count(),
            "Social Connections": session.query(SocialCircle).count(),
            "Interests": session.query(InterestsHobbies).count(),
        }

        session.close()

        typer.echo("📊 Database Status:")
        for table, count in counts.items():
            typer.echo(f"  {table}: {count}")

        # Check understanding score
    from user.user_profiling.user_profile_daemon import UnderstandingScoreCalculator

        calculator = UnderstandingScoreCalculator(db)
        score = calculator.calculate_understanding_score()

        typer.echo(f"\n🧠 Understanding Score: {score.overall_score:.2f}")
        typer.echo(f"   Confidence Level: {score.confidence_level:.2f}")
        typer.echo(f"   Data Completeness: {score.data_completeness:.2f}")

        if score.strengths:
            typer.echo(f"   Strengths: {', '.join(score.strengths)}")
        if score.improvement_areas:
            typer.echo(f"   Improvement Areas: {', '.join(score.improvement_areas)}")

    except Exception as e:
        typer.echo(f"❌ Status check failed: {e}")
        raise typer.Exit(1)


@app.command()
def briefing(
    db_path: Optional[str] = typer.Option(None, help="Database path"),
    date: Optional[str] = typer.Option(
        None, help="Date (YYYY-MM-DD format, default: today)"
    ),
    save: bool = typer.Option(False, help="Save briefing to file"),
):
    """Generate daily briefing"""
    typer.echo("📰 Generating daily briefing...")

    async def _generate_briefing():
        try:
            db = UserProfileDatabase(db_path)
            generator = DailyBriefingGenerator(db)

            target_date = datetime.fromisoformat(date) if date else datetime.now()
            briefing = await generator.generate_daily_briefing(target_date)

            # Display briefing
            typer.echo(f"\n📅 Daily Briefing for {target_date.strftime('%Y-%m-%d')}")
            typer.echo("=" * 50)

            if "summary" in briefing:
                typer.echo(f"📋 Summary: {briefing['summary']}")
                typer.echo()

            sections = briefing.get("sections", {})

            # Calendar section
            if "calendar" in sections:
                cal = sections["calendar"]
                typer.echo(
                    f"📅 Calendar: {cal.get('today_events_count', 0)} events today"
                )
                if cal.get("next_event"):
                    next_event = cal["next_event"]
                    typer.echo(
                        f"   Next: {next_event['title']} at {next_event['time']}"
                    )
                if cal.get("predicted_stress_level", 0) > 0.5:
                    typer.echo(
                        f"   ⚠️  Stress Level: {cal['predicted_stress_level']:.1f}"
                    )
                typer.echo()

            # Academic section
            if "education" in sections:
                edu = sections["education"]
                typer.echo(
                    f"🎓 Academic: {edu.get('active_courses', 0)} active courses"
                )
                deadlines = edu.get("upcoming_deadlines", [])
                if deadlines:
                    typer.echo(f"   📝 {len(deadlines)} upcoming deadlines")
                    for deadline in deadlines[:3]:
                        urgency = deadline.get("urgency", "medium")
                        emoji = (
                            "🔴"
                            if urgency == "critical"
                            else "🟡" if urgency == "high" else "🟢"
                        )
                        typer.echo(
                            f"     {emoji} {deadline['title']} ({deadline['days_until']} days)"
                        )
                typer.echo()

            # Personal section
            if "personal" in sections:
                personal = sections["personal"]
                typer.echo(
                    f"👤 Personal: {personal.get('total_social_connections', 0)} connections"
                )
                interests = personal.get("top_interests", [])
                if interests:
                    typer.echo(f"   🎯 Top Interests: {', '.join(interests[:3])}")
                typer.echo()

            # Recommendations
            if "recommendations" in sections:
                recs = sections["recommendations"]
                if recs:
                    typer.echo("💡 Recommendations:")
                    for rec in recs:
                        typer.echo(f"   • {rec}")
                    typer.echo()

            # Save to file if requested
            if save:
                briefing_dir = Path("user_profiling/briefings")
                briefing_dir.mkdir(exist_ok=True)

                briefing_file = (
                    briefing_dir / f"briefing_{target_date.strftime('%Y%m%d')}.json"
                )
                with open(briefing_file, "w") as f:
                    json.dump(briefing, f, indent=2, default=str)

                typer.echo(f"💾 Briefing saved to: {briefing_file}")

        except Exception as e:
            typer.echo(f"❌ Briefing generation failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_generate_briefing())


@app.command()
def persona(
    db_path: Optional[str] = typer.Option(None, help="Database path"),
    show_context: bool = typer.Option(False, help="Show detailed context analysis"),
):
    """Show current persona state"""
    typer.echo("🎭 Analyzing persona state...")

    async def _analyze_persona():
        try:
            db = UserProfileDatabase(db_path)
            persona_engine = PersonaEngine(db)

            # Get persona summary
            summary = await persona_engine.get_persona_summary()

            typer.echo(f"🎭 Current Persona: {summary.get('current_mode', 'Unknown')}")
            typer.echo(f"⚡ Energy Level: {summary.get('energy_level', 0):.2f}")
            typer.echo(f"😰 Stress Level: {summary.get('stress_level', 0):.2f}")

            context_factors = summary.get("context_factors", {})
            if context_factors:
                typer.echo("\n📊 Context Factors:")
                for factor, value in context_factors.items():
                    if isinstance(value, (int, float)):
                        typer.echo(f"  {factor}: {value:.2f}")
                    else:
                        typer.echo(f"  {factor}: {value}")

            if show_context:
                behavior = await persona_engine.get_current_behavior_settings()
                if behavior:
                    typer.echo("\n⚙️  Behavior Settings:")
                    for setting, value in behavior.items():
                        if isinstance(value, (int, float)):
                            typer.echo(f"  {setting}: {value:.2f}")
                        else:
                            typer.echo(f"  {setting}: {value}")

        except Exception as e:
            typer.echo(f"❌ Persona analysis failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_analyze_persona())


@app.command()
def scan(
    db_path: Optional[str] = typer.Option(None, help="Database path"),
    scan_type: str = typer.Option("mini", help="Scan type: mini or full"),
):
    """Run profile scan"""
    typer.echo(f"🔍 Running {scan_type} profile scan...")

    async def _run_scan():
        try:
            config = DaemonConfig()
            daemon = UserProfileDaemon(config, db_path)

            if scan_type == "full":
                await daemon._run_full_scan()
                typer.echo("✅ Full scan completed")
            else:
                await daemon._run_mini_scan()
                typer.echo("✅ Mini scan completed")

        except Exception as e:
            typer.echo(f"❌ Scan failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_run_scan())


@app.command()
def daemon(
    db_path: Optional[str] = typer.Option(None, help="Database path"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path"),
):
    """Start user profiling daemon"""
    typer.echo("🤖 Starting user profiling daemon...")

    async def _run_daemon():
        try:
            config = DaemonConfig()

            # Load config from file if provided
            if config_file and os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)

            daemon = UserProfileDaemon(config, db_path)

            typer.echo("✅ Daemon started. Press Ctrl+C to stop.")
            await daemon.start_daemon()

        except KeyboardInterrupt:
            typer.echo("\n🛑 Daemon stopped by user")
        except Exception as e:
            typer.echo(f"❌ Daemon failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_run_daemon())


@app.command()
def export(
    db_path: Optional[str] = typer.Option(None, help="Database path"),
    output_file: str = typer.Option("profile_export.json", help="Output file path"),
    include_sensitive: bool = typer.Option(False, help="Include sensitive data"),
):
    """Export user profile data"""
    typer.echo("📦 Exporting user profile data...")

    try:
        db = UserProfileDatabase(db_path)
        session = db.get_session()

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "include_sensitive": include_sensitive,
            "data": {},
        }

        # Export data from each table
    from user.user_profiling.database.user_profile_schema import (
            IdentityCore,
            ContactInfo,
            Calendar,
            EducationProfile,
            FinancialStatus,
            SocialCircle,
            InterestsHobbies,
        )

        # Identity data
        identity_records = (
            session.query(IdentityCore).filter(IdentityCore.is_active == True).all()
        )
        export_data["data"]["identity"] = [
            {
                "full_name": r.full_name,
                "birth_date": r.birth_date.isoformat() if r.birth_date else None,
                "age": r.age,
                "primary_language": r.primary_language,
                "primary_location": r.primary_location,
            }
            for r in identity_records
        ]

        # Calendar data (last 30 days)
        if include_sensitive:
            recent_date = datetime.now() - timedelta(days=30)
            calendar_records = (
                session.query(Calendar)
                .filter(Calendar.created_at >= recent_date, Calendar.is_active == True)
                .all()
            )

            export_data["data"]["calendar"] = [
                {
                    "title": r.title,
                    "event_type": r.event_type,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "importance_score": r.importance_score,
                    "stress_level": r.stress_level,
                }
                for r in calendar_records
            ]

        # Interests
        interest_records = (
            session.query(InterestsHobbies)
            .filter(InterestsHobbies.is_active == True)
            .all()
        )
        export_data["data"]["interests"] = [
            {
                "title": r.title,
                "interest_type": r.interest_type,
                "engagement_level": r.engagement_level,
                "category": r.category,
            }
            for r in interest_records
        ]

        session.close()

        # Save export
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        typer.echo(f"✅ Profile data exported to: {output_file}")

        # Show summary
        typer.echo("\n📊 Export Summary:")
        for category, data in export_data["data"].items():
            typer.echo(f"  {category}: {len(data)} records")

    except Exception as e:
        typer.echo(f"❌ Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def test_integration():
    """Test integration with main Vega2.0 system"""
    typer.echo("🧪 Testing integration with Vega2.0...")

    try:
        # Test core imports
        try:
            from core.config import get_config
            from core.db import log_conversation
            from core.llm import query_llm

            typer.echo("✅ Vega2.0 core components accessible")
        except ImportError as e:
            typer.echo(f"⚠️  Vega2.0 core not accessible: {e}")

        # Test user profiling imports
        try:
            from user.user_profiling.vega_integration import (
                startup_user_profiling,
                ContextualIntelligenceEngine,
                UserProfileManager,
            )

            typer.echo("✅ User profiling components accessible")
        except ImportError as e:
            typer.echo(f"❌ User profiling not accessible: {e}")

        # Test database
        try:
            db = UserProfileDatabase()
            session = db.get_session()
            session.close()
            typer.echo("✅ User profiling database accessible")
        except Exception as e:
            typer.echo(f"❌ Database not accessible: {e}")

        typer.echo("\n🎯 Integration test completed")

    except Exception as e:
        typer.echo(f"❌ Integration test failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
