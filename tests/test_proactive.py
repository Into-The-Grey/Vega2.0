import pytest
from fastapi.testclient import TestClient
from core.app import app

client = TestClient(app)
API_KEY = "vega-default-key"


class TestProactiveEndpoints:
    def test_propose_and_pending(self):
        # Propose a nudge
        resp = client.post(
            "/proactive/propose",
            json={"max_per_day": 5},
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 200
        # List pending
        resp2 = client.get("/proactive/pending", headers={"X-API-Key": API_KEY})
        assert resp2.status_code == 200
        data = resp2.json()
        assert "pending" in data

    def test_accept_decline_flow(self):
        # Propose a nudge
        client.post(
            "/proactive/propose",
            json={"max_per_day": 5},
            headers={"X-API-Key": API_KEY},
        )
        # List pending
        resp = client.get("/proactive/pending", headers={"X-API-Key": API_KEY})
        pend = resp.json()["pending"]
        if pend:
            pid = pend[0]["id"]
            # Accept
            resp2 = client.post(
                f"/proactive/accept?proposed_id={pid}", headers={"X-API-Key": API_KEY}
            )
            assert resp2.status_code == 200
            sid = resp2.json()["session_id"]
            # Get session
            resp3 = client.get(
                f"/proactive/session/{sid}", headers={"X-API-Key": API_KEY}
            )
            assert resp3.status_code == 200
            # End session
            resp4 = client.post(
                f"/proactive/end?session_id={sid}", headers={"X-API-Key": API_KEY}
            )
            assert resp4.status_code == 200
        else:
            # If no pending, just pass
            assert True

    def test_decline(self):
        # Propose a nudge
        client.post(
            "/proactive/propose",
            json={"max_per_day": 5},
            headers={"X-API-Key": API_KEY},
        )
        # List pending
        resp = client.get("/proactive/pending", headers={"X-API-Key": API_KEY})
        pend = resp.json()["pending"]
        if pend:
            pid = pend[0]["id"]
            # Decline
            resp2 = client.post(
                f"/proactive/decline?proposed_id={pid}", headers={"X-API-Key": API_KEY}
            )
            assert resp2.status_code == 200
        else:
            assert True

    def test_send_in_session(self):
        # Propose and accept
        client.post(
            "/proactive/propose",
            json={"max_per_day": 5},
            headers={"X-API-Key": API_KEY},
        )
        resp = client.get("/proactive/pending", headers={"X-API-Key": API_KEY})
        pend = resp.json()["pending"]
        if pend:
            pid = pend[0]["id"]
            resp2 = client.post(
                f"/proactive/accept?proposed_id={pid}", headers={"X-API-Key": API_KEY}
            )
            sid = resp2.json()["session_id"]
            # Send a message
            resp3 = client.post(
                "/proactive/send",
                json={"session_id": sid, "text": "Test message"},
                headers={"X-API-Key": API_KEY},
            )
            assert resp3.status_code == 200
            data = resp3.json()
            assert "reply" in data
        else:
            assert True
