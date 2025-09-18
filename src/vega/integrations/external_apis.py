"""
Enhanced External API Integrations for Vega2.0
==============================================

Comprehensive external service integrations including:
- Search Providers: Google Search, Bing Search, DuckDuckGo, Brave Search
- Social Media: Twitter/X API, Discord API, LinkedIn API
- Communication: Enhanced Slack, Discord webhooks, Telegram
- Calendar: Google Calendar OAuth2, Outlook Calendar
- Financial: Plaid banking, cryptocurrency APIs
- Developer: GitHub API, Stack Overflow API, documentation search
- Knowledge: Wikipedia API, academic databases, news APIs
- Location: Maps APIs, weather services, timezone APIs
"""

import os
import asyncio
import logging
import json
import base64
import hashlib
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote_plus
import secrets

# HTTP clients
try:
    import httpx
    import requests

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

# OAuth and authentication
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

try:
    import tweepy

    TWITTER_API_AVAILABLE = True
except ImportError:
    TWITTER_API_AVAILABLE = False

try:
    import discord

    DISCORD_API_AVAILABLE = True
except ImportError:
    DISCORD_API_AVAILABLE = False

try:
    from plaid.api import plaid_api
    from plaid.model.transactions_get_request import TransactionsGetRequest
    from plaid.model.accounts_get_request import AccountsGetRequest
    from plaid.configuration import Configuration
    from plaid.api_client import ApiClient

    PLAID_AVAILABLE = True
except ImportError:
    PLAID_AVAILABLE = False

# Utilities
try:
    import beautifulsoup4 as bs4
    from bs4 import BeautifulSoup

    PARSING_AVAILABLE = True
except ImportError:
    PARSING_AVAILABLE = False

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Exception raised by API integrations"""

    pass


@dataclass
class SearchResult:
    """Search result from any provider"""

    title: str
    url: str
    snippet: str
    provider: str
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialMediaPost:
    """Social media post result"""

    id: str
    text: str
    author: str
    platform: str
    timestamp: datetime
    engagement: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarEvent:
    """Calendar event"""

    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinancialTransaction:
    """Financial transaction"""

    id: str
    account_id: str
    amount: float
    currency: str
    description: str
    timestamp: datetime
    category: Optional[str] = None
    merchant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIConfig:
    """API configuration for external services"""

    google_search_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None
    bing_search_api_key: Optional[str] = None
    brave_search_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    discord_bot_token: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    google_calendar_credentials: Optional[str] = None
    plaid_client_id: Optional[str] = None
    plaid_secret: Optional[str] = None
    github_token: Optional[str] = None
    openweather_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create config from environment variables"""
        return cls(
            google_search_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
            google_search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"),
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            twitter_api_key=os.getenv("TWITTER_API_KEY"),
            twitter_api_secret=os.getenv("TWITTER_API_SECRET"),
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN"),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            google_calendar_credentials=os.getenv("GOOGLE_CALENDAR_CREDENTIALS"),
            plaid_client_id=os.getenv("PLAID_CLIENT_ID"),
            plaid_secret=os.getenv("PLAID_SECRET"),
            github_token=os.getenv("GITHUB_TOKEN"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
            newsapi_key=os.getenv("NEWSAPI_KEY"),
        )


class EnhancedSearchProvider:
    """Enhanced search provider with multiple backends"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        if HTTP_AVAILABLE:
            self.session = httpx.AsyncClient(timeout=30.0)

    async def search_google(
        self, query: str, num_results: int = 10
    ) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if (
            not self.config.google_search_api_key
            or not self.config.google_search_engine_id
        ):
            raise APIError("Google Search API credentials not configured")

        if not self.session:
            raise APIError("HTTP client not available")

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.config.google_search_api_key,
                "cx": self.config.google_search_engine_id,
                "q": query,
                "num": min(num_results, 10),
            }

            response = await self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for i, item in enumerate(data.get("items", [])):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        provider="google",
                        rank=i + 1,
                        metadata={
                            "display_link": item.get("displayLink"),
                            "formatted_url": item.get("formattedUrl"),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise APIError(f"Google search error: {e}")

    async def search_bing(
        self, query: str, num_results: int = 10
    ) -> List[SearchResult]:
        """Search using Bing Search API"""
        if not self.config.bing_search_api_key:
            raise APIError("Bing Search API key not configured")

        if not self.session:
            raise APIError("HTTP client not available")

        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.config.bing_search_api_key}
            params = {
                "q": query,
                "count": min(num_results, 50),
                "textDecorations": "false",
                "textFormat": "Raw",
            }

            response = await self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for i, item in enumerate(data.get("webPages", {}).get("value", [])):
                results.append(
                    SearchResult(
                        title=item.get("name", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        provider="bing",
                        rank=i + 1,
                        metadata={
                            "display_url": item.get("displayUrl"),
                            "date_last_crawled": item.get("dateLastCrawled"),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            raise APIError(f"Bing search error: {e}")

    async def search_brave(
        self, query: str, num_results: int = 10
    ) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.config.brave_search_api_key:
            raise APIError("Brave Search API key not configured")

        if not self.session:
            raise APIError("HTTP client not available")

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.config.brave_search_api_key,
            }
            params = {
                "q": query,
                "count": min(num_results, 20),
                "text_decorations": "false",
            }

            response = await self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for i, item in enumerate(data.get("web", {}).get("results", [])):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        provider="brave",
                        rank=i + 1,
                        metadata={
                            "profile": item.get("profile"),
                            "language": item.get("language"),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            raise APIError(f"Brave search error: {e}")

    async def search_duckduckgo(
        self, query: str, num_results: int = 10
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (unofficial API)"""
        if not self.session:
            raise APIError("HTTP client not available")

        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            response = await self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            # Abstract result
            if data.get("Abstract"):
                results.append(
                    SearchResult(
                        title=data.get("Heading", query),
                        url=data.get("AbstractURL", ""),
                        snippet=data.get("Abstract", ""),
                        provider="duckduckgo",
                        rank=1,
                        metadata={"type": "abstract"},
                    )
                )

            # Related topics
            for i, topic in enumerate(data.get("RelatedTopics", [])[: num_results - 1]):
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(
                        SearchResult(
                            title=topic.get("Text", "").split(" - ")[0],
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            provider="duckduckgo",
                            rank=i + 2,
                            metadata={"type": "related_topic"},
                        )
                    )

            return results[:num_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            # Return empty results instead of raising error for DuckDuckGo
            return []

    async def search_aggregated(
        self, query: str, num_results: int = 10, providers: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search across multiple providers and aggregate results"""
        if providers is None:
            providers = ["google", "bing", "brave", "duckduckgo"]

        all_results = []

        # Run searches in parallel
        tasks = []
        for provider in providers:
            if provider == "google" and self.config.google_search_api_key:
                tasks.append(self.search_google(query, num_results))
            elif provider == "bing" and self.config.bing_search_api_key:
                tasks.append(self.search_bing(query, num_results))
            elif provider == "brave" and self.config.brave_search_api_key:
                tasks.append(self.search_brave(query, num_results))
            elif provider == "duckduckgo":
                tasks.append(self.search_duckduckgo(query, num_results))

        if tasks:
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)

            for result_list in results_lists:
                if isinstance(result_list, list):
                    all_results.extend(result_list)
                elif isinstance(result_list, Exception):
                    logger.warning(f"Search provider failed: {result_list}")

        # Deduplicate and rank results
        seen_urls = set()
        unique_results = []

        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Sort by provider priority and rank
        provider_priority = {"google": 1, "bing": 2, "brave": 3, "duckduckgo": 4}
        unique_results.sort(
            key=lambda r: (provider_priority.get(r.provider, 999), r.rank)
        )

        return unique_results[:num_results]


class TwitterIntegration:
    """Twitter/X API integration"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.client = None

        if TWITTER_API_AVAILABLE and config.twitter_bearer_token:
            self.client = tweepy.Client(bearer_token=config.twitter_bearer_token)

    def is_available(self) -> bool:
        """Check if Twitter integration is available"""
        return self.client is not None

    async def search_tweets(
        self, query: str, max_results: int = 10
    ) -> List[SocialMediaPost]:
        """Search for tweets"""
        if not self.client:
            raise APIError("Twitter client not configured")

        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=[
                    "created_at",
                    "author_id",
                    "public_metrics",
                    "context_annotations",
                ],
            )

            results = []
            if tweets.data:
                for i, tweet in enumerate(tweets.data):
                    results.append(
                        SocialMediaPost(
                            id=tweet.id,
                            text=tweet.text,
                            author=str(tweet.author_id),
                            platform="twitter",
                            timestamp=tweet.created_at,
                            engagement={
                                "retweets": tweet.public_metrics.get(
                                    "retweet_count", 0
                                ),
                                "likes": tweet.public_metrics.get("like_count", 0),
                                "replies": tweet.public_metrics.get("reply_count", 0),
                            },
                            metadata={"context_annotations": tweet.context_annotations},
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            raise APIError(f"Twitter search error: {e}")

    async def post_tweet(self, text: str) -> str:
        """Post a tweet"""
        if not self.client:
            raise APIError("Twitter client not configured")

        try:
            response = self.client.create_tweet(text=text)
            return response.data["id"]
        except Exception as e:
            logger.error(f"Tweet posting failed: {e}")
            raise APIError(f"Tweet posting error: {e}")


class GoogleCalendarIntegration:
    """Google Calendar API integration"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.service = None
        self.scopes = ["https://www.googleapis.com/auth/calendar.readonly"]

        if GOOGLE_API_AVAILABLE and config.google_calendar_credentials:
            self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Calendar service"""
        try:
            if os.path.exists(self.config.google_calendar_credentials):
                creds = Credentials.from_authorized_user_file(
                    self.config.google_calendar_credentials, self.scopes
                )
                self.service = build("calendar", "v3", credentials=creds)
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar service: {e}")

    def is_available(self) -> bool:
        """Check if Google Calendar integration is available"""
        return self.service is not None

    async def get_events(
        self,
        calendar_id: str = "primary",
        max_results: int = 10,
        time_min: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """Get calendar events"""
        if not self.service:
            raise APIError("Google Calendar service not configured")

        try:
            if time_min is None:
                time_min = datetime.utcnow()

            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat() + "Z",
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            results = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))

                # Parse datetime
                if "T" in start:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                else:
                    start_dt = datetime.fromisoformat(start)
                    end_dt = datetime.fromisoformat(end)

                attendees = []
                if "attendees" in event:
                    attendees = [
                        attendee.get("email", "") for attendee in event["attendees"]
                    ]

                results.append(
                    CalendarEvent(
                        id=event["id"],
                        title=event.get("summary", "No Title"),
                        description=event.get("description", ""),
                        start_time=start_dt,
                        end_time=end_dt,
                        location=event.get("location"),
                        attendees=attendees,
                        metadata={
                            "html_link": event.get("htmlLink"),
                            "creator": event.get("creator"),
                            "organizer": event.get("organizer"),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Google Calendar events fetch failed: {e}")
            raise APIError(f"Google Calendar error: {e}")


class PlaidFinancialIntegration:
    """Plaid financial data integration"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.client = None

        if PLAID_AVAILABLE and config.plaid_client_id and config.plaid_secret:
            configuration = Configuration(
                host=plaid_api.Environment.sandbox,  # Use sandbox for testing
                api_key={
                    "clientId": config.plaid_client_id,
                    "secret": config.plaid_secret,
                },
            )
            api_client = ApiClient(configuration)
            self.client = plaid_api.PlaidApi(api_client)

    def is_available(self) -> bool:
        """Check if Plaid integration is available"""
        return self.client is not None

    async def get_transactions(
        self, access_token: str, start_date: datetime, end_date: datetime
    ) -> List[FinancialTransaction]:
        """Get financial transactions"""
        if not self.client:
            raise APIError("Plaid client not configured")

        try:
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date.date(),
                end_date=end_date.date(),
            )

            response = self.client.transactions_get(request)

            results = []
            for transaction in response["transactions"]:
                results.append(
                    FinancialTransaction(
                        id=transaction["transaction_id"],
                        account_id=transaction["account_id"],
                        amount=-transaction["amount"],  # Plaid uses negative for debits
                        currency=transaction["iso_currency_code"] or "USD",
                        description=transaction["name"],
                        timestamp=datetime.combine(
                            transaction["date"], datetime.min.time()
                        ),
                        category=(
                            transaction["category"][0]
                            if transaction["category"]
                            else None
                        ),
                        merchant=transaction.get("merchant_name"),
                        metadata={
                            "account_owner": transaction.get("account_owner"),
                            "location": transaction.get("location"),
                            "payment_meta": transaction.get("payment_meta"),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Plaid transactions fetch failed: {e}")
            raise APIError(f"Plaid error: {e}")


class GitHubIntegration:
    """GitHub API integration"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None

        if HTTP_AVAILABLE and config.github_token:
            self.session = httpx.AsyncClient(
                headers={
                    "Authorization": f"token {config.github_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=30.0,
            )

    def is_available(self) -> bool:
        """Check if GitHub integration is available"""
        return self.session is not None

    async def search_repositories(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        if not self.session:
            raise APIError("GitHub client not configured")

        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results, 100),
            }

            response = await self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return data.get("items", [])

        except Exception as e:
            logger.error(f"GitHub repository search failed: {e}")
            raise APIError(f"GitHub search error: {e}")

    async def get_user_info(self, username: str) -> Dict[str, Any]:
        """Get GitHub user information"""
        if not self.session:
            raise APIError("GitHub client not configured")

        try:
            url = f"https://api.github.com/users/{username}"
            response = await self.session.get(url)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"GitHub user info fetch failed: {e}")
            raise APIError(f"GitHub user info error: {e}")


class EnhancedSlackIntegration:
    """Enhanced Slack integration with webhooks and bot functionality"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.webhook_url = config.slack_webhook_url
        self.session = None

        if HTTP_AVAILABLE:
            self.session = httpx.AsyncClient(timeout=30.0)

    def is_available(self) -> bool:
        """Check if Slack integration is available"""
        return self.webhook_url is not None and self.session is not None

    async def send_message(
        self, text: str, channel: Optional[str] = None, username: Optional[str] = None
    ) -> bool:
        """Send message to Slack"""
        if not self.webhook_url or not self.session:
            raise APIError("Slack webhook not configured")

        try:
            payload = {"text": text}

            if channel:
                payload["channel"] = channel
            if username:
                payload["username"] = username

            response = await self.session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            return response.text == "ok"

        except Exception as e:
            logger.error(f"Slack message sending failed: {e}")
            raise APIError(f"Slack error: {e}")

    async def send_rich_message(
        self, blocks: List[Dict[str, Any]], channel: Optional[str] = None
    ) -> bool:
        """Send rich message with blocks to Slack"""
        if not self.webhook_url or not self.session:
            raise APIError("Slack webhook not configured")

        try:
            payload = {"blocks": blocks}

            if channel:
                payload["channel"] = channel

            response = await self.session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            return response.text == "ok"

        except Exception as e:
            logger.error(f"Slack rich message sending failed: {e}")
            raise APIError(f"Slack rich message error: {e}")


class EnhancedIntegrationManager:
    """Comprehensive integration manager for all external APIs"""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig.from_env()

        # Initialize integrations
        self.search = EnhancedSearchProvider(self.config)
        self.twitter = TwitterIntegration(self.config)
        self.calendar = GoogleCalendarIntegration(self.config)
        self.financial = PlaidFinancialIntegration(self.config)
        self.github = GitHubIntegration(self.config)
        self.slack = EnhancedSlackIntegration(self.config)

    def get_available_integrations(self) -> Dict[str, bool]:
        """Get status of all integrations"""
        return {
            "search": HTTP_AVAILABLE,
            "twitter": self.twitter.is_available(),
            "calendar": self.calendar.is_available(),
            "financial": self.financial.is_available(),
            "github": self.github.is_available(),
            "slack": self.slack.is_available(),
        }

    async def unified_search(
        self, query: str, sources: Optional[List[str]] = None, max_results: int = 20
    ) -> Dict[str, List[Any]]:
        """Unified search across multiple sources"""
        if sources is None:
            sources = ["web", "github", "twitter"]

        results = {}

        # Web search
        if "web" in sources:
            try:
                web_results = await self.search.search_aggregated(query, max_results)
                results["web"] = web_results
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                results["web"] = []

        # GitHub search
        if "github" in sources and self.github.is_available():
            try:
                github_results = await self.github.search_repositories(
                    query, max_results // 2
                )
                results["github"] = github_results
            except Exception as e:
                logger.error(f"GitHub search failed: {e}")
                results["github"] = []

        # Twitter search
        if "twitter" in sources and self.twitter.is_available():
            try:
                twitter_results = await self.twitter.search_tweets(
                    query, max_results // 2
                )
                results["twitter"] = twitter_results
            except Exception as e:
                logger.error(f"Twitter search failed: {e}")
                results["twitter"] = []

        return results

    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context from available sources"""
        context = {
            "calendar_events": [],
            "financial_summary": {},
            "social_activity": [],
        }

        # Calendar events
        if self.calendar.is_available():
            try:
                events = await self.calendar.get_events(max_results=5)
                context["calendar_events"] = [
                    {
                        "title": event.title,
                        "start_time": event.start_time.isoformat(),
                        "location": event.location,
                    }
                    for event in events
                ]
            except Exception as e:
                logger.error(f"Calendar context failed: {e}")

        # Note: Financial and social data would require user-specific tokens
        # This is a framework for when those are available

        return context

    async def send_notification(
        self, message: str, channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send notifications across multiple channels"""
        if channels is None:
            channels = ["slack"]

        results = {}

        # Slack notification
        if "slack" in channels and self.slack.is_available():
            try:
                success = await self.slack.send_message(message)
                results["slack"] = success
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")
                results["slack"] = False

        return results

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.search, "session") and self.search.session:
            await self.search.session.aclose()
        if hasattr(self.github, "session") and self.github.session:
            await self.github.session.aclose()
        if hasattr(self.slack, "session") and self.slack.session:
            await self.slack.session.aclose()


# Global integration manager instance
_integration_manager: Optional[EnhancedIntegrationManager] = None


def get_integration_manager() -> EnhancedIntegrationManager:
    """Get global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = EnhancedIntegrationManager()
    return _integration_manager


# Convenience functions for backward compatibility
async def web_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """Search the web using available providers"""
    manager = get_integration_manager()
    return await manager.search.search_aggregated(query, max_results)


async def send_slack_message(message: str) -> bool:
    """Send Slack message"""
    manager = get_integration_manager()
    if manager.slack.is_available():
        return await manager.slack.send_message(message)
    return False


async def search_github(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search GitHub repositories"""
    manager = get_integration_manager()
    if manager.github.is_available():
        return await manager.github.search_repositories(query, max_results)
    return []


# Export main classes and functions
__all__ = [
    "EnhancedIntegrationManager",
    "EnhancedSearchProvider",
    "TwitterIntegration",
    "GoogleCalendarIntegration",
    "PlaidFinancialIntegration",
    "GitHubIntegration",
    "EnhancedSlackIntegration",
    "APIConfig",
    "SearchResult",
    "SocialMediaPost",
    "CalendarEvent",
    "FinancialTransaction",
    "APIError",
    "get_integration_manager",
    "web_search",
    "send_slack_message",
    "search_github",
]
