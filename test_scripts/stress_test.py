#!/usr/bin/env python3
"""
Vega2.0 Stress Test Suite - Find the Breaking Points
Run actual tests, not theoretical monitoring.
"""

import asyncio
import time
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class StressTest:
    def __init__(self):
        self.results = []
        self.failures = []

    def log(self, test_name: str, status: str, details: str = ""):
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time(),
        }
        self.results.append(result)

        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")

        if status == "FAIL":
            self.failures.append(result)

    def run_cli_command(
        self, prompt: str, timeout: int = 30
    ) -> tuple[bool, str, float]:
        """Run a CLI command and measure response time"""
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

            if result.returncode == 0:
                # Filter out warning/info lines
                output_lines = [
                    line
                    for line in result.stdout.split("\n")
                    if not line.startswith("✅")
                    and not line.startswith("⚠️")
                    and not line.startswith("WARNING")
                    and not line.startswith("Optional")
                    and line.strip()
                ]
                output = "\n".join(output_lines)
                return True, output, elapsed
            else:
                return False, result.stderr, elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return False, f"Timeout after {timeout}s", elapsed
        except Exception as e:
            elapsed = time.time() - start
            return False, str(e), elapsed

    def test_basic_response(self):
        """Test 1: Can it respond at all?"""
        success, output, elapsed = self.run_cli_command("Say 'test'")

        if success and len(output) > 0:
            self.log("Basic Response", "PASS", f"{elapsed:.2f}s")
        else:
            self.log("Basic Response", "FAIL", f"No output or error: {output}")

    def test_response_speed(self):
        """Test 2: Response time for simple query"""
        success, output, elapsed = self.run_cli_command("What is 2+2?")

        if success:
            if elapsed < 5:
                self.log(
                    "Response Speed (Simple)", "PASS", f"{elapsed:.2f}s (excellent)"
                )
            elif elapsed < 10:
                self.log(
                    "Response Speed (Simple)", "PASS", f"{elapsed:.2f}s (acceptable)"
                )
            else:
                self.log("Response Speed (Simple)", "WARN", f"{elapsed:.2f}s (slow)")
        else:
            self.log("Response Speed (Simple)", "FAIL", output)

    def test_long_response(self):
        """Test 3: Can it handle long response generation?"""
        prompt = "Write a detailed explanation of how neural networks work, covering backpropagation, activation functions, and optimization."
        success, output, elapsed = self.run_cli_command(prompt, timeout=60)

        if success:
            word_count = len(output.split())
            if word_count > 50:
                self.log("Long Response", "PASS", f"{elapsed:.2f}s, {word_count} words")
            else:
                self.log(
                    "Long Response", "WARN", f"Response too short: {word_count} words"
                )
        else:
            self.log("Long Response", "FAIL", output)

    def test_complex_prompt(self):
        """Test 4: Complex reasoning task"""
        prompt = "If I have 3 apples and give 2 to my friend, then buy 5 more, how many do I have?"
        success, output, elapsed = self.run_cli_command(prompt)

        if success:
            # Check if answer contains "6"
            if "6" in output or "six" in output.lower():
                self.log("Complex Reasoning", "PASS", f"{elapsed:.2f}s")
            else:
                self.log(
                    "Complex Reasoning",
                    "WARN",
                    f"Answer may be incorrect: {output[:100]}",
                )
        else:
            self.log("Complex Reasoning", "FAIL", output)

    def test_rapid_fire(self):
        """Test 5: Multiple quick requests"""
        print("\n🔥 Rapid fire test (5 quick requests)...")

        prompts = ["Hi", "What's 5+5?", "Name a color", "Say hello", "What's today?"]

        times = []
        failures = 0

        for i, prompt in enumerate(prompts, 1):
            print(f"  Request {i}/5...", end=" ", flush=True)
            success, output, elapsed = self.run_cli_command(prompt, timeout=15)
            times.append(elapsed)

            if success:
                print(f"✅ {elapsed:.2f}s")
            else:
                print(f"❌ Failed")
                failures += 1

        avg_time = sum(times) / len(times)

        if failures == 0:
            self.log("Rapid Fire (5x)", "PASS", f"Avg: {avg_time:.2f}s, All succeeded")
        elif failures < 3:
            self.log(
                "Rapid Fire (5x)", "WARN", f"{failures}/5 failed, Avg: {avg_time:.2f}s"
            )
        else:
            self.log("Rapid Fire (5x)", "FAIL", f"{failures}/5 failed")

    def test_special_characters(self):
        """Test 6: Special characters and encoding"""
        prompt = "Repeat: Hello! @#$%^&*()_+ 日本語 émojis: 🚀🎉"
        success, output, elapsed = self.run_cli_command(prompt)

        if success:
            self.log("Special Characters", "PASS", f"{elapsed:.2f}s")
        else:
            self.log("Special Characters", "FAIL", output)

    def test_empty_prompt(self):
        """Test 7: Edge case - empty prompt"""
        success, output, elapsed = self.run_cli_command("")

        # Empty prompt should either handle gracefully or fail fast
        if elapsed < 2:
            self.log("Empty Prompt", "PASS", f"Failed fast: {elapsed:.2f}s")
        else:
            self.log("Empty Prompt", "WARN", f"Took too long to fail: {elapsed:.2f}s")

    def test_very_long_prompt(self):
        """Test 8: Very long input"""
        # Generate a long prompt
        prompt = "Count from 1 to 100: " + ", ".join(str(i) for i in range(1, 101))
        success, output, elapsed = self.run_cli_command(prompt, timeout=45)

        if success:
            self.log("Very Long Prompt", "PASS", f"{elapsed:.2f}s")
        else:
            self.log("Very Long Prompt", "FAIL", output)

    def test_memory_check(self):
        """Test 9: Check if previous conversation is remembered"""
        print("\n💭 Testing conversation memory...")

        # First message
        success1, _, _ = self.run_cli_command(
            "Remember this: my favorite color is blue"
        )
        time.sleep(1)

        # Second message asking about it
        success2, output2, elapsed = self.run_cli_command("What's my favorite color?")

        if success1 and success2:
            if "blue" in output2.lower():
                self.log("Memory/Context", "PASS", f"Remembered previous context")
            else:
                self.log("Memory/Context", "FAIL", "Did not remember previous context")
        else:
            self.log("Memory/Context", "FAIL", "One or both requests failed")

    def test_concurrent_load(self):
        """Test 10: Concurrent requests (async)"""
        print("\n⚡ Concurrent load test (3 simultaneous)...")

        async def run_concurrent():
            prompts = ["What is Python?", "Explain AI", "Name 3 colors"]

            tasks = []
            start = time.time()

            for prompt in prompts:
                # Run them via subprocess but track them
                task = asyncio.create_subprocess_exec(
                    "/usr/bin/python3",
                    "main.py",
                    "cli",
                    "chat",
                    prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=Path(__file__).parent,
                )
                tasks.append(task)

            # Start all
            procs = await asyncio.gather(*tasks)

            # Wait for completion
            results = []
            for proc in procs:
                stdout, stderr = await proc.communicate()
                results.append((proc.returncode, stdout.decode()))

            elapsed = time.time() - start

            successes = sum(1 for rc, _ in results if rc == 0)
            return successes, elapsed

        try:
            successes, elapsed = asyncio.run(run_concurrent())

            if successes == 3:
                self.log(
                    "Concurrent Load (3x)",
                    "PASS",
                    f"{elapsed:.2f}s total, all succeeded",
                )
            elif successes > 0:
                self.log(
                    "Concurrent Load (3x)",
                    "WARN",
                    f"{successes}/3 succeeded in {elapsed:.2f}s",
                )
            else:
                self.log("Concurrent Load (3x)", "FAIL", "All failed")
        except Exception as e:
            self.log("Concurrent Load (3x)", "FAIL", str(e))

    def run_all_tests(self):
        """Run all stress tests"""
        print("=" * 60)
        print("🔥 VEGA 2.0 STRESS TEST SUITE")
        print("=" * 60)
        print("Finding breaking points so we can fix them...\n")

        tests = [
            self.test_basic_response,
            self.test_response_speed,
            self.test_long_response,
            self.test_complex_prompt,
            self.test_rapid_fire,
            self.test_special_characters,
            self.test_empty_prompt,
            self.test_very_long_prompt,
            self.test_memory_check,
            self.test_concurrent_load,
        ]

        for i, test in enumerate(tests, 1):
            print(f"\n[Test {i}/{len(tests)}]")
            try:
                test()
            except Exception as e:
                self.log(test.__doc__ or "Unknown", "FAIL", f"Exception: {str(e)}")
            time.sleep(0.5)  # Brief pause between tests

        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 STRESS TEST SUMMARY")
        print("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        warned = sum(1 for r in self.results if r["status"] == "WARN")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")

        print(f"\n✅ Passed: {passed}/{total}")
        print(f"⚠️  Warnings: {warned}/{total}")
        print(f"❌ Failed: {failed}/{total}")

        if self.failures:
            print("\n🔴 FAILURES TO ADDRESS:")
            for f in self.failures:
                print(f"\n  • {f['test']}")
                print(f"    {f['details']}")

        if failed == 0 and warned == 0:
            print("\n🎉 All tests passed! System is solid.")
        elif failed == 0:
            print("\n✨ No failures, but some warnings to check.")
        else:
            print(f"\n⚠️  Found {failed} issues to fix.")

        # Save results
        results_file = Path(__file__).parent / "stress_test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Full results saved to: {results_file}")


if __name__ == "__main__":
    tester = StressTest()
    tester.run_all_tests()
