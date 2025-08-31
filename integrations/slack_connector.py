"""
slack_connector.py - Slack webhook integration stub

- Posts a message to a Slack Incoming Webhook URL
- Keep this simple and optional; failures are non-fatal
- Extend pattern to support Discord, email, generic webhooks
"""

from __future__ import annotations

import json
from typing import Optional

import requests


def send_slack_message(webhook_url: Optional[str], text: str) -> bool:
    """Post a basic text message via Slack webhook.

    Returns True if request sent (status 2xx), False if no webhook configured or on error.
    """
    if not webhook_url:
        return False
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.ok
    except Exception:
        return False
