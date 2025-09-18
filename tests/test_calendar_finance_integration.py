import os
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..user.user_profiling.collectors import calendar_sync, finance_monitor

@pytest.mark.asyncio
def test_google_calendar_env_loading(monkeypatch):
    monkeypatch.setenv("GOOGLE_CALENDAR_CREDENTIALS", "test_credentials.json")
    monkeypatch.setenv("GOOGLE_CALENDAR_SCOPES", '["https://www.googleapis.com/auth/calendar.readonly"]')
    config = calendar_sync.CalendarConfig()
    assert config.google_credentials_file == "test_credentials.json"
    assert config.google_scopes == ["https://www.googleapis.com/auth/calendar.readonly"]


def test_plaid_env_loading(monkeypatch):
    monkeypatch.setenv("PLAID_CLIENT_ID", "test_id")
    monkeypatch.setenv("PLAID_SECRET", "test_secret")
    config = finance_monitor.FinancialConfig()
    assert config.enable_plaid_integration is True


def test_calendar_config_defaults():
    config = calendar_sync.CalendarConfig()
    assert config.sync_past_days == 30
    assert config.sync_future_days == 90


def test_financial_config_defaults():
    config = finance_monitor.FinancialConfig()
    assert config.privacy_mode is True
    assert config.trend_analysis_days == 90

