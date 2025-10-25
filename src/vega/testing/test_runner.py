from __future__ import annotations

import os
import json
import time
import socket
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime

import subprocess


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_ms: int
    details: str = ""


@dataclass
class TestCase:
    name: str
    category: str
    func: Callable[[], Tuple[bool, str]]

    def run(self) -> TestResult:
        start = time.perf_counter()
        try:
            passed, details = self.func()
        except Exception as e:
            passed = False
            details = f"Exception: {e}"
        duration_ms = int((time.perf_counter() - start) * 1000)
        return TestResult(self.name, self.category, passed, duration_ms, details)


def _http_get(
    url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 3.0
) -> Tuple[int, str]:
    try:
        import httpx  # lightweight, async-capable HTTP client

        r = httpx.get(url, headers=headers or {}, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        # Fallback to urllib
        try:
            import urllib.request

            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="ignore")
        except Exception as e2:
            raise RuntimeError(f"HTTP GET failed: {e} | Fallback: {e2}")


def _http_post_json(
    url: str,
    payload: dict,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 5.0,
) -> Tuple[int, str]:
    try:
        import httpx

        r = httpx.post(url, json=payload, headers=headers or {}, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        try:
            import urllib.request
            import json as _json

            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", **(headers or {})},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="ignore")
        except Exception as e2:
            raise RuntimeError(f"HTTP POST failed: {e} | Fallback: {e2}")


# -------------------- Individual test implementations --------------------


def _test_server_health() -> Tuple[bool, str]:
    base = "http://127.0.0.1:8000"
    # /healthz
    code, body = _http_get(f"{base}/healthz")
    if code != 200 or "ok" not in body:
        return False, f"/healthz failed: {code} {body[:120]}"
    # /livez
    code, _ = _http_get(f"{base}/livez")
    if code != 200:
        return False, f"/livez failed: {code}"
    # /readyz
    code, _ = _http_get(f"{base}/readyz")
    if code != 200:
        return False, f"/readyz failed: {code}"
    return True, "healthz/livez/readyz OK"


def _test_server_chat_security() -> Tuple[bool, str]:
    base = "http://127.0.0.1:8000"
    payload = {"prompt": "ping", "stream": False}
    code, _ = _http_post_json(f"{base}/chat", payload)
    if code == 401:
        return True, "chat without API key correctly rejected (401)"
    return False, f"chat without API key should be 401, got {code}"


def _test_server_chat_basic() -> Tuple[bool, str]:
    base = "http://127.0.0.1:8000"
    payload = {"prompt": "hello", "stream": False}
    headers = {"X-API-Key": "vega-default-key"}
    code, body = _http_post_json(f"{base}/chat", payload, headers=headers)
    if code != 200:
        return False, f"chat failed: {code} {body[:120]}"
    if "Echo:" not in body and "response" not in body:
        return False, f"unexpected chat response body: {body[:120]}"
    return True, "chat with API key OK"


def _test_systemd_services() -> Tuple[bool, str]:
    # Validate that systemctl exists and services are active
    def _is_active(name: str) -> Tuple[bool, str]:
        try:
            res = subprocess.run(
                ["systemctl", "is-active", name], capture_output=True, text=True
            )
            active = res.returncode == 0 and res.stdout.strip() == "active"
            return active, res.stdout.strip() or res.stderr.strip()
        except FileNotFoundError:
            return False, "systemctl not available"

    ok1, out1 = _is_active("vega")
    ok2, out2 = _is_active("vega-daemon")
    if not ok1 and "not available" in out1:
        return False, "systemctl not available on this system"
    if ok1 and ok2:
        return True, "vega and vega-daemon are active"
    return False, f"service states: vega={out1}, vega-daemon={out2}"


def _test_daemon_state_and_health() -> Tuple[bool, str]:
    from ..daemon.system_manager import VegaSystemManager

    home = os.path.expanduser("~")
    state_path = os.path.join(home, ".vega", "system_state.json")
    if not os.path.exists(state_path):
        return False, f"state file missing: {state_path}"
    try:
        import json as _json

        with open(state_path, "r", encoding="utf-8") as f:
            _json.load(f)
    except Exception as e:
        return False, f"state file unreadable: {e}"

    mgr = VegaSystemManager()
    health = mgr.monitor_health()
    required = {
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "server_running",
        "suggestions",
    }
    if not required.issubset(health.keys()):
        return (
            False,
            f"health keys missing: expected {required}, got {set(health.keys())}",
        )
    return True, "daemon state and health OK"


def _test_database_roundtrip() -> Tuple[bool, str]:
    from ..core.db import log_conversation, get_history

    pid = log_conversation(
        "test prompt", "test response", source="test", session_id="test-session"
    )
    hist = get_history(limit=5)
    if not any(r.get("id") == pid for r in hist):
        return False, "logged conversation not found in recent history"
    return True, "database log/get_history OK"


def _test_dataset_builder() -> Tuple[bool, str]:
    from ..datasets.prepare_dataset import build_dataset
    from pathlib import Path

    samples_dir = Path(__file__).parent.parent / "datasets" / "samples"
    out = build_dataset(str(samples_dir))
    if not os.path.exists(out):
        return False, f"output dataset not found: {out}"
    # ensure file has at least one line
    with open(out, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line.strip():
            return False, "output dataset is empty"
    return True, f"dataset built at {out}"


def _test_integrations_search() -> Tuple[bool, str]:
    try:
        from ..integrations.search import web_search
    except Exception as e:
        return False, f"cannot import search: {e}"
    # This may return [] if package/network unavailable; treat as soft pass
    try:
        res = web_search("Vega2.0 AI", max_results=2)
        if isinstance(res, list):
            return True, f"web_search returned {len(res)} results (empty acceptable)"
        return False, f"web_search returned non-list: {type(res)}"
    except Exception as e:
        return False, f"web_search raised: {e}"


def _test_training_config_loads() -> Tuple[bool, str]:
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent / "training" / "config.yaml"
    if not cfg_path.exists():
        return False, f"training config not found: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required_keys = {"model_name", "training_strategy", "output_dir", "train_file"}
    if not required_keys.issubset(set(cfg.keys())):
        return False, f"training config missing keys: {required_keys - set(cfg.keys())}"
    return True, "training config loads with required keys"


# -------------------- Category assembly and NL mapping --------------------


def _build_test_cases() -> Dict[str, List[TestCase]]:
    return {
        "server": [
            TestCase("server_health", "server", _test_server_health),
            TestCase("server_chat_security", "server", _test_server_chat_security),
            TestCase("server_chat_basic", "server", _test_server_chat_basic),
        ],
        "daemon": [
            TestCase("systemd_services", "daemon", _test_systemd_services),
            TestCase("daemon_state_health", "daemon", _test_daemon_state_and_health),
        ],
        "database": [
            TestCase("db_roundtrip", "database", _test_database_roundtrip),
        ],
        "datasets": [
            TestCase("dataset_builder", "datasets", _test_dataset_builder),
        ],
        "integrations": [
            TestCase("search_basic", "integrations", _test_integrations_search),
        ],
        "training": [
            TestCase("training_config", "training", _test_training_config_loads),
        ],
        # Security is implicitly covered by server_chat_security; include category alias
        "security": [
            TestCase("chat_requires_api_key", "security", _test_server_chat_security),
        ],
    }


AVAILABLE_CATEGORIES = sorted(_build_test_cases().keys())


_NL_MAP = {
    "server": ["server", "api", "fastapi", "endpoint", "health", "chat", "http"],
    "daemon": ["daemon", "systemd", "service", "monitor", "vega-daemon"],
    "database": ["db", "database", "sqlite", "conversation", "history", "log"],
    "datasets": ["dataset", "data", "prepare", "jsonl", "samples"],
    "integrations": ["integration", "search", "osint", "web"],
    "training": ["train", "training", "lora", "config", "peft"],
    "security": ["security", "auth", "api key", "authentication", "authorization"],
}


def infer_categories(phrase: str) -> List[str]:
    p = phrase.lower()
    cats: List[str] = []
    for cat, keywords in _NL_MAP.items():
        if any(k in p for k in keywords):
            cats.append(cat)
    # Fallback: if no keywords matched, default to 'server'
    return cats or ["server"]


def _append_comment(text: str, category: str = "TESTING") -> None:
    try:
        from ..daemon.system_manager import VegaSystemManager

        mgr = VegaSystemManager()
        mgr.add_comment(text, category)
    except Exception:
        # Fallback: write directly to ~/VEGA_COMMENTS.txt
        try:
            home = os.path.expanduser("~")
            comments = os.path.join(home, "VEGA_COMMENTS.txt")
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            with open(comments, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"[{ts}] {category}\n")
                f.write(text + "\n\n")
        except Exception:
            pass


def run_tests(
    phrase: Optional[str] = None,
    categories: Optional[List[str]] = None,
    interactive: bool = False,
    destructive: bool = False,
) -> Dict[str, any]:
    """
    Run selected tests and return a structured report dict.

    - phrase: natural language intent (maps to categories)
    - categories: explicit list of categories to run
    - interactive: if True, prompt user for category selection (ignored here; used by CLI)
    - destructive: reserved for potentially destructive tests (not used in current set)
    """
    cases_map = _build_test_cases()

    selected: List[str] = []
    if categories:
        for c in categories:
            if c in cases_map:
                selected.append(c)
    elif phrase:
        selected = infer_categories(phrase)
    else:
        selected = list(cases_map.keys())  # run all by default

    tests: List[TestCase] = []
    for cat in selected:
        tests.extend(cases_map.get(cat, []))

    results: List[TestResult] = []
    for t in tests:
        results.append(t.run())

    # Summarize
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "selected_categories": selected,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [asdict(r) for r in results],
    }

    # Persist report under ~/.vega/test_reports
    try:
        home = os.path.expanduser("~")
        out_dir = os.path.join(home, ".vega", "test_reports")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summary["report_path"] = out_path
    except Exception:
        pass

    # Append a short comment
    try:
        _append_comment(
            text=(
                f"Test run: {summary['passed']}/{summary['total']} passed, "
                f"categories={','.join(selected)}"
            ),
            category="TESTING",
        )
    except Exception:
        pass

    return summary
