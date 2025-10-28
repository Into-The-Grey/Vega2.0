"""
slack_connector.py - Slack webhook integration stub

- Posts a message to a Slack Incoming Webhook URL
- Keep this simple and optional; failures are non-fatal
- Extend pattern to support Discord, email, generic webhooks
- Uses shared HTTP client for connection pooling
"""

from __future__ import annotations

import json
from typing import Optional


async def send_slack_message(webhook_url: Optional[str], text: str) -> bool:
    """Post a basic text message via Slack webhook using shared HTTP client.

    Returns True if request sent (status 2xx), False if no webhook configured or on error.
    """
    if not webhook_url:
        return False

    try:
        # Use shared HTTP client from resource manager
        from ..core.resource_manager import get_resource_manager

        manager = await get_resource_manager()
        client = manager.get_http_client_direct()

        r = await client.post(webhook_url, json={"text": text}, timeout=10.0)
        return r.is_success
    except Exception:
        return False
