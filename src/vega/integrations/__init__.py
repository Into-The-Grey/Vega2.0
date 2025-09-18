"""
Enhanced Integrations Module for Vega2.0
========================================

Comprehensive external service integrations with multiple providers
and intelligent fallback support.
"""

import logging
from typing import Dict, List, Optional, Union, Any

# Import enhanced integrations
try:
    from .integrations_enhanced import (
        EnhancedIntegrationManager,
        EnhancedSearchProvider,
        TwitterIntegration,
        GoogleCalendarIntegration,
        PlaidFinancialIntegration,
        GitHubIntegration,
        EnhancedSlackIntegration,
        APIConfig,
        SearchResult,
        SocialMediaPost,
        CalendarEvent,
        FinancialTransaction,
        APIError,
        get_integration_manager as get_enhanced_manager,
        web_search as enhanced_web_search,
        send_slack_message as enhanced_slack_message,
        search_github as enhanced_github_search,
    )

    ENHANCED_AVAILABLE = True
except ImportError as e:
    ENHANCED_AVAILABLE = False
    logging.warning(f"Enhanced integrations not available: {e}")

# Import legacy integrations for fallback
from .search import web_search as legacy_web_search, image_search as legacy_image_search
from .slack_connector import send_slack_message as legacy_slack_message
from .osint import (
    dns_lookup,
    ssl_cert_info,
    http_headers as get_http_headers,
    tcp_scan as port_scan,
)
from .fetch import fetch_text as fetch_webpage

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Main integration manager with enhanced capabilities and legacy fallback
    """

    def __init__(self, use_enhanced: bool = True):
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self._enhanced_manager = None

        if self.use_enhanced:
            try:
                self._enhanced_manager = get_enhanced_manager()
            except Exception as e:
                logger.warning(f"Enhanced integrations not available: {e}")
                self.use_enhanced = False

    async def search_web(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search the web using best available provider"""
        if self.use_enhanced and self._enhanced_manager:
            try:
                results = await enhanced_web_search(query, max_results)
                return [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "provider": r.provider,
                        "rank": r.rank,
                    }
                    for r in results
                ]
            except Exception as e:
                logger.error(f"Enhanced web search failed: {e}")

        # Fallback to legacy search
        try:
            return legacy_web_search(query, max_results)
        except Exception as e:
            logger.error(f"Legacy web search failed: {e}")
            return []

    async def search_images(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for images"""
        if self.use_enhanced and self._enhanced_manager:
            # Enhanced image search would be implemented here
            logger.info("Enhanced image search not yet implemented, using legacy")

        # Use legacy image search
        try:
            return legacy_image_search(query, max_results)
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    async def send_slack_notification(
        self, message: str, channel: Optional[str] = None
    ) -> bool:
        """Send Slack notification"""
        if self.use_enhanced and self._enhanced_manager:
            try:
                return await enhanced_slack_message(message)
            except Exception as e:
                logger.error(f"Enhanced Slack notification failed: {e}")

        # Fallback to legacy Slack
        try:
            import os

            webhook_url = os.getenv("SLACK_WEBHOOK_URL")
            return legacy_slack_message(webhook_url, message)
        except Exception as e:
            logger.error(f"Legacy Slack notification failed: {e}")
            return False

    def get_available_integrations(self) -> Dict[str, bool]:
        """Get status of available integrations"""
        if self.use_enhanced and self._enhanced_manager:
            return self._enhanced_manager.get_available_integrations()
        else:
            return {
                "search": True,  # Legacy search always available
                "slack": True,  # Legacy slack always available
                "osint": True,  # OSINT tools always available
                "fetch": True,  # Web fetching always available
                "twitter": False,
                "calendar": False,
                "financial": False,
                "github": False,
            }


# Global integration manager
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager() -> IntegrationManager:
    """Get global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager


# Convenience functions (backward compatibility)
async def web_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search the web"""
    manager = get_integration_manager()
    return await manager.search_web(query, max_results)


async def image_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search for images"""
    manager = get_integration_manager()
    return await manager.search_images(query, max_results)


async def send_slack_message(message: str) -> bool:
    """Send Slack message"""
    manager = get_integration_manager()
    return await manager.send_slack_notification(message)


def get_available_integrations() -> Dict[str, bool]:
    """Get available integrations status"""
    manager = get_integration_manager()
    return manager.get_available_integrations()


# Re-export useful functions from legacy modules
# Note: These are aliased from the actual function names in the modules

# Export main classes and functions
__all__ = [
    "IntegrationManager",
    "get_integration_manager",
    "web_search",
    "image_search",
    "send_slack_message",
    "get_available_integrations",
    "dns_lookup",
    "ssl_cert_info",
    "get_http_headers",
    "port_scan",
    "fetch_webpage",
]
