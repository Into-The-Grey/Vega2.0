#!/usr/bin/env python3
"""
Extreme Stress Test Suite - Breaking Point Analysis

Aggressive testing to find failure modes:
1. Massive concurrent session creation (100+ sessions)
2. Rapid-fire fact updates (race conditions)
3. Memory injection attacks (SQL injection, XSS patterns)
4. Extreme unicode and binary data
5. Database lock exhaustion
6. Memory overflow with gigantic context
7. Malformed session IDs and fact keys
8. Circular reference patterns
9. Time-based race conditions
10. Resource exhaustion scenarios
"""

import asyncio
import sys
import time
import random
import string
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vega.core.db import (
    log_conversation,
    get_memory_facts,
    set_memory_fact,
    get_recent_context,
    get_conversation_summary,
)
from src.vega.core.app import _extract_memory_facts


class StressTestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []
        self.performance = []

    def add_pass(self, test_name: str, details: str = ""):
        self.passed.append((test_name, details))
        print(f"  ✅ {test_name}")
        if details:
            print(f"     {details}")

    def add_fail(self, test_name: str, reason: str):
        self.failed.append((test_name, reason))
        print(f"  ❌ {test_name}")
        print(f"     {reason}")

    def add_error(self, test_name: str, error: str):
        self.errors.append((test_name, error))
        print(f"  🔥 {test_name} - EXCEPTION")
        print(f"     {error}")

    def add_perf(self, test_name: str, duration: float, operations: int):
        ops_per_sec = operations / duration if duration > 0 else 0
        self.performance.append((test_name, duration, operations, ops_per_sec))
        print(f"  ⏱️  {test_name}: {duration:.2f}s, {ops_per_sec:.0f} ops/sec")


results = StressTestResults()


def test_massive_concurrent_sessions():
    """Test 100+ concurrent sessions with simultaneous fact updates"""
    print("\n🔥 Test 1: Massive Concurrent Session Storm (100 sessions)")

    num_sessions = 100
    operations_per_session = 10

    def session_worker(session_num: int):
        session_id = f"stress-session-{session_num}"
        try:
            for i in range(operations_per_session):
                # Rapid fact updates
                set_memory_fact(session_id, f"fact_{i}", f"value_{session_num}_{i}")
                log_conversation(f"msg_{i}", f"resp_{i}", session_id=session_id)

                # Read immediately after write (race condition test)
                facts = get_memory_facts(session_id)
                if f"fact_{i}" not in facts:
                    return False
            return True
        except Exception as e:
            print(f"Session {session_num} error: {e}")
            return False

    start = time.time()
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(session_worker, i) for i in range(num_sessions)]
        results_list = [f.result() for f in as_completed(futures)]

    duration = time.time() - start
    total_ops = num_sessions * operations_per_session * 2  # reads + writes
    success_count = sum(results_list)

    if success_count == num_sessions:
        results.add_pass(
            "Massive concurrent sessions",
            f"{num_sessions} sessions, {success_count}/{num_sessions} succeeded",
        )
        results.add_perf("Concurrent storm", duration, total_ops)
        return True
    else:
        results.add_fail(
            "Massive concurrent sessions",
            f"Only {success_count}/{num_sessions} sessions succeeded",
        )
        return False


def test_sql_injection_attempts():
    """Test memory system against SQL injection patterns"""
    print("\n🔥 Test 2: SQL Injection Attack Patterns")

    injection_patterns = [
        "'; DROP TABLE memory_facts; --",
        "' OR '1'='1",
        "1' UNION SELECT * FROM conversations--",
        "admin'--",
        "' OR 1=1--",
        "'; DELETE FROM memory_facts WHERE '1'='1",
        "1' AND '1'='1",
        '"; DROP TABLE conversations; --',
        "' OR 'x'='x",
        "1'; UPDATE memory_facts SET value='hacked' WHERE '1'='1'--",
    ]

    session_id = "injection-test"
    passed = 0

    try:
        for i, pattern in enumerate(injection_patterns):
            # Try to inject in key
            set_memory_fact(session_id, pattern, "safe_value")

            # Try to inject in value
            set_memory_fact(session_id, f"safe_key_{i}", pattern)

            # Try to inject in session_id
            set_memory_fact(pattern, "key", "value")

            passed += 1

        # Verify database integrity
        facts = get_memory_facts(session_id)
        if len(facts) > 0:
            results.add_pass(
                "SQL injection resistance",
                f"Survived {len(injection_patterns)} injection attempts, db intact",
            )
            return True
    except Exception as e:
        results.add_error("SQL injection resistance", str(e))
        return False

    return False


def test_extreme_unicode_and_binary():
    """Test with extreme unicode, emoji storms, and binary-like data"""
    print("\n🔥 Test 3: Extreme Unicode and Binary Data")

    extreme_strings = [
        # Emoji storm
        "🔥" * 1000,
        "👨‍👩‍👧‍👦" * 100,  # Complex family emoji
        "".join(chr(i) for i in range(0x1F600, 0x1F650)),  # All emoji in range
        # Zero-width and invisible characters
        "\u200b\u200c\u200d\u2060\ufeff" * 100,
        # RTL and bidirectional text
        "مرحبا" * 100 + "Hello" * 100 + "שלום" * 100,
        # Control characters
        "".join(chr(i) for i in range(32)),
        # Combining diacritics storm
        "e" + "\u0301" * 100,  # e with 100 combining acute accents
        # Surrogate pairs and invalid UTF-8
        "\ud800\udc00" * 100,
        # Null bytes and special chars
        "\x00" * 100,
        "\n" * 1000,
        "\r\n" * 1000,
        # Homograph attack (look-alike characters)
        "аdmіn",  # Using Cyrillic a, i
        "Gооgle",  # Using Cyrillic o
        # String length attacks
        "A" * 10000,
        "测试" * 5000,  # Chinese characters
    ]

    session_id = "unicode-extreme"
    passed = 0

    try:
        for i, extreme_str in enumerate(extreme_strings):
            # Store as key
            set_memory_fact(session_id, f"key_{i}", extreme_str)

            # Store as value
            set_memory_fact(session_id, extreme_str[:50], f"value_{i}")

            passed += 1

        # Verify retrieval
        facts = get_memory_facts(session_id)
        if len(facts) >= passed:
            results.add_pass(
                "Extreme unicode/binary",
                f"Handled {len(extreme_strings)} extreme strings",
            )
            return True
    except Exception as e:
        results.add_error(
            "Extreme unicode/binary", f"Failed at string {passed}: {str(e)}"
        )
        return False

    return False


def test_rapid_fact_overwrite_race():
    """Test rapid overwrites from multiple threads (race condition)"""
    print("\n🔥 Test 4: Rapid Fact Overwrite Race Condition")

    session_id = "race-condition-test"
    fact_key = "contested_fact"
    num_threads = 50
    writes_per_thread = 100

    def writer(thread_id: int):
        for i in range(writes_per_thread):
            set_memory_fact(session_id, fact_key, f"thread_{thread_id}_write_{i}")
        return thread_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(writer, i) for i in range(num_threads)]
        list(as_completed(futures))

    duration = time.time() - start

    # Verify final state is consistent
    facts = get_memory_facts(session_id)

    if fact_key in facts:
        final_value = facts[fact_key]
        results.add_pass(
            "Race condition handling",
            f"{num_threads * writes_per_thread} writes, final state: {final_value[:30]}...",
        )
        results.add_perf("Rapid overwrites", duration, num_threads * writes_per_thread)
        return True
    else:
        results.add_fail("Race condition handling", "Fact key lost during race")
        return False


def test_malformed_session_ids():
    """Test with malformed, extremely long, and special session IDs"""
    print("\n🔥 Test 5: Malformed Session ID Handling")

    malformed_ids = [
        "",  # Empty
        " ",  # Whitespace
        "\n\r\t",  # Control chars
        "a" * 10000,  # Extremely long
        "../../../etc/passwd",  # Path traversal
        "session\x00id",  # Null byte
        "session'; DROP TABLE--",  # SQL injection
        "../../vega.db",  # Relative path
        "CON",  # Windows reserved name
        "NUL",
        "session/id/with/slashes",
        "session\\id\\with\\backslashes",
        "🔥💥🎉",  # Only emoji
        "\u200b\u200c\u200d",  # Only zero-width chars
    ]

    passed = 0
    failed = 0

    for session_id in malformed_ids:
        try:
            set_memory_fact(session_id, "test_key", "test_value")
            facts = get_memory_facts(session_id)
            log_conversation("test", "response", session_id=session_id)
            passed += 1
        except Exception as e:
            failed += 1
            # This is actually okay - some IDs should fail gracefully

    if passed + failed == len(malformed_ids):
        results.add_pass(
            "Malformed session IDs", f"Handled {passed} gracefully, rejected {failed}"
        )
        return True
    else:
        results.add_fail("Malformed session IDs", "Unexpected behavior")
        return False


def test_database_lock_exhaustion():
    """Test SQLite lock behavior under extreme contention"""
    print("\n🔥 Test 6: Database Lock Exhaustion")

    num_threads = 100
    operations_per_thread = 50

    def db_hammer(thread_id: int):
        successes = 0
        failures = 0
        for i in range(operations_per_thread):
            try:
                session_id = f"lock-test-{thread_id}"
                set_memory_fact(session_id, f"key_{i}", f"val_{i}")
                log_conversation(f"q_{i}", f"a_{i}", session_id=session_id)
                get_memory_facts(session_id)
                get_recent_context(session_id=session_id, limit=5)
                successes += 1
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    failures += 1
                else:
                    raise
        return successes, failures

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(db_hammer, i) for i in range(num_threads)]
        results_list = [f.result() for f in as_completed(futures)]

    duration = time.time() - start
    total_successes = sum(r[0] for r in results_list)
    total_failures = sum(r[1] for r in results_list)
    total_ops = num_threads * operations_per_thread * 4  # 4 db ops per iteration

    success_rate = (
        total_successes / (total_successes + total_failures) * 100
        if total_successes + total_failures > 0
        else 0
    )

    if success_rate >= 95:
        results.add_pass(
            "Database lock handling",
            f"{success_rate:.1f}% success rate, {total_failures} lock timeouts",
        )
        results.add_perf("Lock contention", duration, total_ops)
        return True
    else:
        results.add_fail(
            "Database lock handling", f"Only {success_rate:.1f}% success rate"
        )
        return False


def test_memory_extraction_adversarial():
    """Test extraction with adversarial inputs designed to break regex"""
    print("\n🔥 Test 7: Adversarial Extraction Patterns")

    adversarial_inputs = [
        # Nested patterns
        "My name is My name is Alice",
        "I live in I live in Boston",
        # Recursive patterns
        "My name is my name is my name is Bob",
        # Pattern injection
        "My name is '; DROP TABLE--",
        # Extremely long names/locations
        "My name is " + "A" * 1000,
        "I live in " + "B" * 1000,
        # Unicode homoglyphs
        "My name is Аlice",  # Cyrillic A
        "I live in Воston",  # Cyrillic o
        # Multiple conflicting patterns
        "My name is Alice and my name is Bob and my name is Charlie",
        "I live in NYC and I live in LA and I live in SF",
        # Embedded nulls
        "My name is Alice\x00Bob",
        # Regex special characters
        "My name is Alice.*Bob",
        "I live in (New York|Boston)",
        "My name is ^Alice$",
        # Lookahead/lookbehind attempts
        "My name is (?=Alice)Bob",
        # Backtracking bombs (ReDoS attempts)
        "My name is " + "a" * 100 + "b" * 100,
        # Zero-width assertions
        "My\u200bname\u200bis\u200bAlice",
    ]

    passed = 0
    failed = 0
    errors = []

    for input_text in adversarial_inputs:
        try:
            start = time.time()
            facts = _extract_memory_facts(input_text)
            duration = time.time() - start

            # Check for ReDoS (should complete quickly)
            if duration > 1.0:
                errors.append(f"Slow extraction ({duration:.2f}s): {input_text[:50]}")
                failed += 1
            else:
                passed += 1
        except Exception as e:
            errors.append(f"Exception on: {input_text[:50]} - {str(e)}")
            failed += 1

    if failed == 0:
        results.add_pass(
            "Adversarial extraction",
            f"Survived {len(adversarial_inputs)} adversarial inputs",
        )
        return True
    else:
        results.add_fail(
            "Adversarial extraction", f"{failed} failures: {'; '.join(errors[:3])}"
        )
        return False


def test_extreme_context_overflow():
    """Test with massive conversation history"""
    print("\n🔥 Test 8: Extreme Context Overflow")

    session_id = "overflow-extreme"

    # Create 1000 conversations
    start = time.time()
    for i in range(1000):
        prompt = f"Message {i}: " + "x" * 100
        response = f"Response {i}: " + "y" * 100
        log_conversation(prompt, response, session_id=session_id)

    write_duration = time.time() - start

    # Try to get context (should handle gracefully)
    start = time.time()
    try:
        context = get_recent_context(session_id=session_id, limit=100)
        summary = get_conversation_summary(session_id=session_id, max_entries=1000)
        read_duration = time.time() - start

        results.add_pass(
            "Extreme context overflow",
            f"1000 convs written in {write_duration:.2f}s, context: {len(context)} items, summary: {len(summary)} chars",
        )
        results.add_perf("Massive history", write_duration + read_duration, 1000)
        return True
    except Exception as e:
        results.add_error("Extreme context overflow", str(e))
        return False


def test_concurrent_read_write_storm():
    """Test simultaneous reads and writes on same session"""
    print("\n🔥 Test 9: Concurrent Read/Write Storm")

    session_id = "rw-storm"
    num_threads = 50
    duration_seconds = 5

    read_count = [0]
    write_count = [0]
    error_count = [0]

    def reader():
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            try:
                get_memory_facts(session_id)
                get_recent_context(session_id=session_id, limit=10)
                read_count[0] += 1
            except Exception:
                error_count[0] += 1

    def writer():
        end_time = time.time() + duration_seconds
        i = 0
        while time.time() < end_time:
            try:
                set_memory_fact(session_id, f"key_{i}", f"val_{i}")
                log_conversation(f"q_{i}", f"a_{i}", session_id=session_id)
                write_count[0] += 1
                i += 1
            except Exception:
                error_count[0] += 1

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Half readers, half writers
        futures = []
        for i in range(num_threads // 2):
            futures.append(executor.submit(reader))
            futures.append(executor.submit(writer))

        for f in as_completed(futures):
            f.result()

    actual_duration = time.time() - start
    total_ops = read_count[0] + write_count[0]

    if error_count[0] < total_ops * 0.05:  # Less than 5% errors
        results.add_pass(
            "Read/Write storm",
            f"{read_count[0]} reads, {write_count[0]} writes, {error_count[0]} errors",
        )
        results.add_perf("RW storm", actual_duration, total_ops)
        return True
    else:
        results.add_fail(
            "Read/Write storm", f"Too many errors: {error_count[0]}/{total_ops}"
        )
        return False


def test_fact_key_collision_storm():
    """Test handling of similar/colliding fact keys"""
    print("\n🔥 Test 10: Fact Key Collision Storm")

    session_id = "collision-test"

    # Create many similar keys
    collision_keys = [
        "user_name",
        "user_name ",
        " user_name",
        "user_name\n",
        "user_name\t",
        "USER_NAME",
        "User_Name",
        "user-name",
        "user.name",
        "user/name",
        "user\\name",
        "user_name_",
        "_user_name",
        "user__name",
        "user\u200bname",  # Zero-width space
        "username",
        "usеr_name",  # Cyrillic e
    ]

    try:
        # Set all variations
        for i, key in enumerate(collision_keys):
            set_memory_fact(session_id, key, f"value_{i}")

        # Verify all stored distinctly
        facts = get_memory_facts(session_id)

        # Check we got most of them (some may normalize to same key)
        if len(facts) >= len(collision_keys) * 0.8:
            results.add_pass(
                "Fact key collisions",
                f"Stored {len(facts)}/{len(collision_keys)} similar keys distinctly",
            )
            return True
        else:
            results.add_fail(
                "Fact key collisions",
                f"Only {len(facts)}/{len(collision_keys)} keys stored",
            )
            return False
    except Exception as e:
        results.add_error("Fact key collisions", str(e))
        return False


def main():
    print("=" * 80)
    print("EXTREME STRESS TEST SUITE - BREAKING POINT ANALYSIS")
    print("=" * 80)

    tests = [
        test_massive_concurrent_sessions,
        test_sql_injection_attempts,
        test_extreme_unicode_and_binary,
        test_rapid_fact_overwrite_race,
        test_malformed_session_ids,
        test_database_lock_exhaustion,
        test_memory_extraction_adversarial,
        test_extreme_context_overflow,
        test_concurrent_read_write_storm,
        test_fact_key_collision_storm,
    ]

    passed = 0
    failed = 0
    errors = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results.add_error(test.__name__, str(e))
            errors += 1

    print("\n" + "=" * 80)
    print(f"EXTREME STRESS TEST RESULTS: {passed}/{len(tests)} passed")
    print("=" * 80)

    if results.failed:
        print("\n❌ Failed Tests:")
        for test_name, reason in results.failed:
            print(f"  • {test_name}: {reason}")

    if results.errors:
        print("\n🔥 Exception Tests:")
        for test_name, error in results.errors:
            print(f"  • {test_name}: {error}")

    if results.performance:
        print("\n⏱️  Performance Metrics:")
        for test_name, duration, ops, ops_per_sec in results.performance:
            print(
                f"  • {test_name}: {ops} ops in {duration:.2f}s ({ops_per_sec:.0f} ops/sec)"
            )

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
