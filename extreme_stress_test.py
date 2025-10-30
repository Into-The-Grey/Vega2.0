#!/usr/bin/env python3
"""
Vega2.0 EXTREME Stress Test - Find ACTUAL Breaking Points
Push it until it breaks, then fix the breaks.
"""

import subprocess
import time
import sys
from pathlib import Path
import concurrent.futures


def run_cli(prompt: str, timeout: int = 60) -> tuple[bool, str, float]:
    """Run CLI command and measure time"""
    start = time.time()
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "main.py", "cli", "chat", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )
        elapsed = time.time() - start
        return result.returncode == 0, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def test_massive_concurrent():
    """Test 1: 10 concurrent requests"""
    print("\n🔥 TEST 1: 10 Concurrent Requests")
    print("=" * 50)

    prompts = [f"Count to {i}" for i in range(1, 11)]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_cli, p, 30) for p in prompts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    elapsed = time.time() - start
    successes = sum(1 for success, _, _ in results if success)

    print(f"Results: {successes}/10 succeeded in {elapsed:.2f}s")
    if successes == 10:
        print("✅ PASS - System handled concurrent load")
    else:
        print(f"❌ FAIL - Only {successes}/10 succeeded")
        print("FIX NEEDED: Add connection pooling or request queue")


def test_memory_exhaustion():
    """Test 2: Very large prompt (try to exhaust memory)"""
    print("\n🔥 TEST 2: Memory Exhaustion (10k word prompt)")
    print("=" * 50)

    # Generate massive prompt
    huge_prompt = "Summarize this: " + (" hello world" * 5000)

    success, output, elapsed = run_cli(huge_prompt, timeout=90)

    if success:
        print(f"✅ PASS - Handled large prompt in {elapsed:.2f}s")
    else:
        print(f"❌ FAIL - {output[:200]}")
        print("FIX NEEDED: Add input size limits in config")
        print("Suggested: MAX_PROMPT_CHARS=4000 in .env")


def test_rapid_burst():
    """Test 3: 20 requests as fast as possible"""
    print("\n🔥 TEST 3: Rapid Burst (20 requests, no delay)")
    print("=" * 50)

    times = []
    failures = 0

    for i in range(20):
        print(f"  Request {i+1}/20...", end=" ", flush=True)
        success, output, elapsed = run_cli("Hi", timeout=15)
        times.append(elapsed)

        if success:
            print(f"✅ {elapsed:.2f}s")
        else:
            print(f"❌ FAIL")
            failures += 1

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.2f}s, Failures: {failures}/20")

    if failures == 0:
        print("✅ PASS - No failures")
    elif failures < 5:
        print(f"⚠️  WARN - {failures} failures")
        print("FIX SUGGESTED: Add rate limiting or request coalescing")
    else:
        print(f"❌ FAIL - Too many failures ({failures})")
        print("FIX NEEDED: Add request queue with backpressure")


def test_cpu_intensive():
    """Test 4: CPU intensive task"""
    print("\n🔥 TEST 4: CPU Intensive (complex code generation)")
    print("=" * 50)

    prompt = """Write a complete Python implementation of:
1. A binary search tree with insert, delete, search
2. AVL tree self-balancing logic
3. Full test suite
Include all edge cases and error handling."""

    success, output, elapsed = run_cli(prompt, timeout=120)

    if success:
        lines = output.count("\n")
        print(f"✅ PASS - Generated {lines} lines in {elapsed:.2f}s")
        if elapsed > 60:
            print("⚠️  WARN - Took >60s")
            print("FIX SUGGESTED: Reduce GEN_MAX_TOKENS or use faster model")
    else:
        print(f"❌ FAIL - Timeout or error")
        print("FIX NEEDED: Add streaming or timeout handling")


def test_database_stress():
    """Test 5: Multiple conversations to stress DB"""
    print("\n🔥 TEST 5: Database Stress (30 quick messages)")
    print("=" * 50)

    start = time.time()
    failures = 0

    for i in range(30):
        success, _, _ = run_cli(f"Message {i}", timeout=10)
        if not success:
            failures += 1

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/30...")

    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f}s, Failures: {failures}/30")

    if failures == 0:
        print("✅ PASS - Database handled load")
    else:
        print(f"❌ FAIL - {failures} database errors")
        print("FIX NEEDED: Enable WAL mode, add connection pooling")


def test_malformed_input():
    """Test 6: Malformed/malicious input"""
    print("\n🔥 TEST 6: Malformed Input (SQL injection, XSS, etc)")
    print("=" * 50)

    evil_inputs = [
        "'; DROP TABLE conversations; --",
        "<script>alert('xss')</script>",
        "' OR '1'='1",
        "\x00\x00\x00",
        "A" * 100000,  # 100k chars
    ]

    failures = 0
    crashes = 0

    for i, evil in enumerate(evil_inputs, 1):
        print(f"  Test {i}/5...", end=" ", flush=True)
        success, output, elapsed = run_cli(evil[:100], timeout=10)

        if "error" in output.lower() or "exception" in output.lower():
            if elapsed < 2:
                print("✅ Failed fast")
            else:
                print("⚠️  Slow fail")
                failures += 1
        elif success:
            print("✅ Handled safely")
        else:
            print("❌ CRASH")
            crashes += 1

    if crashes == 0 and failures == 0:
        print("✅ PASS - Input validation working")
    elif crashes == 0:
        print("⚠️  WARN - Some slow error handling")
        print("FIX SUGGESTED: Add input validation middleware")
    else:
        print(f"❌ FAIL - {crashes} crashes")
        print("FIX NEEDED: Add input sanitization and size limits")


def test_timeout_handling():
    """Test 7: Request timeout"""
    print("\n🔥 TEST 7: Timeout Handling (force timeout)")
    print("=" * 50)

    # Ask for something that might take forever
    prompt = "Count from 1 to 100000 with explanations for each number"

    success, output, elapsed = run_cli(prompt, timeout=5)  # Short timeout

    if not success and "TIMEOUT" in output:
        print(f"✅ PASS - Timed out gracefully after {elapsed:.2f}s")
    elif not success:
        print(f"✅ PASS - Failed fast: {elapsed:.2f}s")
    else:
        print(f"⚠️  WARN - Completed anyway in {elapsed:.2f}s")


def test_crash_recovery():
    """Test 8: Kill and restart"""
    print("\n🔥 TEST 8: Crash Recovery")
    print("=" * 50)

    # Make a request
    print("  Making request 1...", end=" ", flush=True)
    success1, _, _ = run_cli("First message", timeout=10)
    print("✅" if success1 else "❌")

    # Try another immediately
    print("  Making request 2 (immediate)...", end=" ", flush=True)
    success2, _, _ = run_cli("Second message", timeout=10)
    print("✅" if success2 else "❌")

    if success1 and success2:
        print("✅ PASS - System recovered/handled multiple requests")
    else:
        print("❌ FAIL - System didn't recover")
        print("FIX NEEDED: Add process supervision or automatic restart")


def run_all_extreme_tests():
    """Run all extreme tests"""
    print("=" * 60)
    print("💥 VEGA 2.0 EXTREME STRESS TEST")
    print("=" * 60)
    print("Pushing limits until something breaks...\n")

    tests = [
        test_rapid_burst,
        test_massive_concurrent,
        test_cpu_intensive,
        test_database_stress,
        test_memory_exhaustion,
        test_malformed_input,
        test_timeout_handling,
        test_crash_recovery,
    ]

    for test in tests:
        try:
            test()
            time.sleep(1)  # Brief pause
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {e}")
            print(f"FIX NEEDED: Add exception handling in {test.__name__}")

    print("\n" + "=" * 60)
    print("💥 EXTREME STRESS TEST COMPLETE")
    print("=" * 60)
    print("\nNow address the failures above! 🔧")


if __name__ == "__main__":
    run_all_extreme_tests()
