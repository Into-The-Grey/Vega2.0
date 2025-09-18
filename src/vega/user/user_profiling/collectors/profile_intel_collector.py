"""
Profile Intelligence Collector
===============================

Autonomous daemon for collecting and enriching user profile data from various sources
including web presence, documents, and digital footprints with privacy controls.
"""

import asyncio
import aiohttp
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
import spacy
from bs4 import BeautifulSoup
import requests_html
from urllib.parse import urljoin, urlparse
import random

from database.user_profile_schema import (
    UserProfileDatabase,
    IdentityCore,
    ContactInfo,
    WebPresence,
    SearchHistory,
    InterestsHobbies,
    SocialCircle,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    """Configuration for intelligence scanning"""

    max_results_per_platform: int = 50
    scan_delay_seconds: int = 2
    max_depth: int = 3
    enable_web_scraping: bool = True
    enable_email_parsing: bool = False  # Requires explicit user permission
    enable_document_parsing: bool = True
    user_agent_rotation: bool = True
    respect_robots_txt: bool = True
    anonymize_browsing: bool = True


class ProfileIntelCollector:
    """Main intelligence collection engine"""

    def __init__(self, db: UserProfileDatabase, config: ScanConfig = None):
        self.db = db
        self.config = config or ScanConfig()
        self.session = None
        self.nlp = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ]

        # Initialize NLP if available
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            logger.warning(
                "spaCy not available. Install with: python -m spacy download en_core_web_sm"
            )

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": random.choice(self.user_agents)},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def get_user_identifiers(self) -> Dict[str, List[str]]:
        """Extract user identifiers from database for searching"""
        session = self.db.get_session()
        try:
            identifiers = {"names": [], "usernames": [], "emails": [], "handles": []}

            # Get names from identity core
            identity_records = (
                session.query(IdentityCore).filter(IdentityCore.is_active == True).all()
            )

            for record in identity_records:
                if record.full_name:
                    identifiers["names"].append(record.full_name)
                if record.aliases:
                    identifiers["names"].extend(record.aliases)

            # Get contact info
            contact_records = (
                session.query(ContactInfo).filter(ContactInfo.is_active == True).all()
            )

            for record in contact_records:
                if record.contact_type == "email":
                    identifiers["emails"].append(record.contact_value)
                elif record.contact_type == "username":
                    identifiers["usernames"].append(record.contact_value)
                elif record.contact_type == "social_handle":
                    identifiers["handles"].append(record.contact_value)

            return identifiers
        finally:
            session.close()

    async def search_platform(self, platform: str, query: str) -> List[Dict]:
        """Search specific platform for user presence"""
        results = []

        platform_configs = {
            "github": {
                "search_url": f"https://api.github.com/search/users?q={query}",
                "profile_url": "https://github.com/{}",
                "requires_auth": False,
            },
            "reddit": {
                "search_url": f"https://www.reddit.com/search.json?q={query}&type=user",
                "profile_url": "https://www.reddit.com/user/{}",
                "requires_auth": False,
            },
            "twitter": {
                "search_url": f"https://twitter.com/search?q={query}",
                "profile_url": "https://twitter.com/{}",
                "requires_auth": True,  # API access limited
            },
        }

        if platform not in platform_configs:
            logger.warning(f"Platform {platform} not supported")
            return results

        config = platform_configs[platform]

        try:
            if platform == "github":
                results = await self._search_github(query)
            elif platform == "reddit":
                results = await self._search_reddit(query)
            # Add more platforms as needed

        except Exception as e:
            logger.error(f"Error searching {platform}: {e}")

        return results

    async def _search_github(self, query: str) -> List[Dict]:
        """Search GitHub for user profiles"""
        results = []

        if not self.session:
            return results

        try:
            url = f"https://api.github.com/search/users?q={query}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    for user in data.get("items", [])[
                        : self.config.max_results_per_platform
                    ]:
                        # Get detailed user info
                        user_url = user["url"]
                        async with self.session.get(user_url) as user_response:
                            if user_response.status == 200:
                                user_data = await user_response.json()

                                result = {
                                    "platform": "github",
                                    "username": user_data.get("login"),
                                    "url": user_data.get("html_url"),
                                    "name": user_data.get("name"),
                                    "bio": user_data.get("bio"),
                                    "location": user_data.get("location"),
                                    "email": user_data.get("email"),
                                    "followers": user_data.get("followers"),
                                    "repositories": user_data.get("public_repos"),
                                    "created_date": user_data.get("created_at"),
                                    "confidence_score": self._calculate_match_confidence(
                                        query, user_data
                                    ),
                                }
                                results.append(result)

                        # Rate limiting
                        await asyncio.sleep(self.config.scan_delay_seconds)

        except Exception as e:
            logger.error(f"GitHub search error: {e}")

        return results

    async def _search_reddit(self, query: str) -> List[Dict]:
        """Search Reddit for user profiles"""
        results = []

        if not self.session:
            return results

        try:
            # Reddit's API approach
            headers = {"User-Agent": "VegaProfiler/1.0"}
            url = f"https://www.reddit.com/search.json?q={query}&type=user&limit=25"

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    for user in data.get("data", {}).get("children", []):
                        user_data = user.get("data", {})

                        result = {
                            "platform": "reddit",
                            "username": user_data.get("name"),
                            "url": f"https://www.reddit.com/user/{user_data.get('name')}",
                            "created_date": datetime.fromtimestamp(
                                user_data.get("created_utc", 0)
                            ).isoformat(),
                            "karma": user_data.get("total_karma", 0),
                            "confidence_score": self._calculate_match_confidence(
                                query, user_data
                            ),
                        }
                        results.append(result)

                await asyncio.sleep(self.config.scan_delay_seconds)

        except Exception as e:
            logger.error(f"Reddit search error: {e}")

        return results

    def _calculate_match_confidence(self, query: str, user_data: Dict) -> float:
        """Calculate confidence score for user match"""
        confidence = 0.0
        query_lower = query.lower()

        # Exact username match
        username = user_data.get("login") or user_data.get("name", "")
        if username.lower() == query_lower:
            confidence += 0.5
        elif query_lower in username.lower():
            confidence += 0.3

        # Name matches
        name = user_data.get("name") or user_data.get("display_name", "")
        if name and query_lower in name.lower():
            confidence += 0.3

        # Bio/description matches
        bio = user_data.get("bio") or user_data.get("description", "")
        if bio and any(word in bio.lower() for word in query_lower.split()):
            confidence += 0.2

        return min(confidence, 1.0)

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using NLP"""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "skills": [],
            "interests": [],
        }

        if not self.nlp or not text:
            return entities

        try:
            doc = self.nlp(text)

            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)

            # Simple keyword extraction for skills/interests
            tech_keywords = [
                "python",
                "javascript",
                "react",
                "node",
                "ml",
                "ai",
                "data science",
                "programming",
                "coding",
                "development",
                "software",
                "web development",
            ]

            text_lower = text.lower()
            for keyword in tech_keywords:
                if keyword in text_lower:
                    entities["skills"].append(keyword)

        except Exception as e:
            logger.error(f"Entity extraction error: {e}")

        return entities

    def parse_documents_in_directory(self, directory: str) -> List[Dict]:
        """Parse documents from input directory"""
        parsed_data = []
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return parsed_data

        for file_path in directory_path.glob("**/*"):
            if file_path.is_file():
                try:
                    content = self._extract_file_content(file_path)
                    if content:
                        entities = self.extract_entities_from_text(content)

                        parsed_data.append(
                            {
                                "file_path": str(file_path),
                                "content_preview": content[:500],
                                "entities": entities,
                                "file_type": file_path.suffix,
                                "parsed_date": datetime.now().isoformat(),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")

        return parsed_data

    def _extract_file_content(self, file_path: Path) -> str:
        """Extract text content from various file types"""
        content = ""

        try:
            if file_path.suffix.lower() in [
                ".txt",
                ".md",
                ".py",
                ".js",
                ".html",
                ".css",
            ]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert JSON to searchable text
                    content = json.dumps(data, indent=2)

            # Add more file type handlers as needed

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

        return content

    async def store_web_presence_data(self, platform_results: List[Dict]):
        """Store web presence data in database"""
        session = self.db.get_session()

        try:
            for result in platform_results:
                # Check if record already exists
                existing = (
                    session.query(WebPresence)
                    .filter(
                        WebPresence.platform == result["platform"],
                        WebPresence.username == result.get("username"),
                        WebPresence.url == result.get("url"),
                    )
                    .first()
                )

                if not existing:
                    web_presence = WebPresence(
                        platform=result["platform"],
                        url=result.get("url"),
                        username=result.get("username"),
                        content_type="profile",
                        content_summary=result.get("bio")
                        or result.get("description", ""),
                        extracted_interests=result.get("interests", []),
                        extracted_skills=result.get("skills", []),
                        extracted_locations=(
                            [result.get("location")] if result.get("location") else []
                        ),
                        confidence_score=result.get("confidence_score", 0.0),
                        is_verified=False,
                        scan_date=datetime.now(),
                    )
                    session.add(web_presence)

            session.commit()
            logger.info(f"Stored {len(platform_results)} web presence records")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing web presence data: {e}")
        finally:
            session.close()

    async def run_full_scan(self) -> Dict[str, Any]:
        """Run complete intelligence collection scan"""
        scan_start = datetime.now()
        results = {
            "scan_id": hashlib.md5(scan_start.isoformat().encode()).hexdigest()[:8],
            "scan_start": scan_start.isoformat(),
            "identifiers_used": [],
            "platforms_scanned": [],
            "documents_parsed": 0,
            "web_records_found": 0,
            "entities_extracted": 0,
            "errors": [],
        }

        try:
            # Get user identifiers
            identifiers = self.get_user_identifiers()
            results["identifiers_used"] = identifiers

            # Scan web platforms
            all_platform_results = []
            platforms = ["github", "reddit"]  # Add more as needed

            for platform in platforms:
                platform_results = []

                # Search using different identifier types
                for name in identifiers.get("names", []):
                    platform_results.extend(await self.search_platform(platform, name))

                for username in identifiers.get("usernames", []):
                    platform_results.extend(
                        await self.search_platform(platform, username)
                    )

                all_platform_results.extend(platform_results)
                results["platforms_scanned"].append(platform)

                # Rate limiting between platforms
                await asyncio.sleep(self.config.scan_delay_seconds * 2)

            # Store web presence data
            if all_platform_results:
                await self.store_web_presence_data(all_platform_results)
                results["web_records_found"] = len(all_platform_results)

            # Parse documents if enabled
            if self.config.enable_document_parsing:
                doc_directory = os.path.join(
                    os.path.dirname(__file__), "..", "..", "input_data", "user_docs"
                )
                parsed_docs = self.parse_documents_in_directory(doc_directory)
                results["documents_parsed"] = len(parsed_docs)

                # Count entities
                total_entities = sum(
                    len(doc.get("entities", {}).get(key, []))
                    for doc in parsed_docs
                    for key in doc.get("entities", {})
                )
                results["entities_extracted"] = total_entities

        except Exception as e:
            error_msg = f"Scan error: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        # Save scan results
        results["scan_end"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - scan_start).total_seconds()

        # Save to intel_scans directory
        scan_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "intel_scans",
            f"{scan_start.strftime('%Y%m%d_%H%M%S')}.json",
        )

        os.makedirs(os.path.dirname(scan_file), exist_ok=True)
        with open(scan_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results


async def run_intelligence_scan(db_path: str = None) -> Dict[str, Any]:
    """Main function to run intelligence collection"""
    db = UserProfileDatabase(db_path)
    config = ScanConfig()

    async with ProfileIntelCollector(db, config) as collector:
        results = await collector.run_full_scan()

    return results


if __name__ == "__main__":
    # Test the intelligence collector
    async def main():
        results = await run_intelligence_scan()
        print("Intelligence Collection Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
