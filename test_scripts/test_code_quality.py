#!/usr/bin/env python3
"""Test system prompt enforcement for production-ready code generation."""

import httpx
import asyncio

API_KEY = "devkey"
API_URL = "http://127.0.0.1:8000/chat"

TEST_PROMPTS = [
    {
        "name": "SQL Injection Scanner",
        "prompt": "Write a Python script that tests a web form for SQL injection vulnerabilities. Include proper error handling and make it production-ready.",
        "expect_keywords": ["import", "def", "try", "except", "requests", "sqlmap"],
    },
    {
        "name": "Port Scanner",
        "prompt": "Create a basic port scanner in Python that can scan common ports on a target host. This is for a lab assignment.",
        "expect_keywords": ["socket", "def", "try", "except", "port", "scan"],
    },
    {
        "name": "Password Strength Checker",
        "prompt": "Write a script to check password strength and detect common patterns. Needs to be reviewed by instructor.",
        "expect_keywords": ["import re", "def", "strength", "pattern", "complexity"],
    },
]

ANTI_PATTERNS = [
    "# TODO",
    "# Placeholder",
    "# Implementation goes here",
    "pass  # Your code here",
    "raise NotImplementedError",
    "... your code ...",
    "dummy",
]


async def test_code_generation():
    """Test that Vega generates production-ready code, not placeholders."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("\n" + "=" * 80)
        print("TESTING: Production-Ready Code Generation")
        print("=" * 80 + "\n")

        for i, test in enumerate(TEST_PROMPTS, 1):
            print(f"\nTest {i}/{len(TEST_PROMPTS)}: {test['name']}")
            print(f"Prompt: {test['prompt'][:100]}...")

            try:
                response = await client.post(
                    API_URL,
                    headers={"X-API-Key": API_KEY},
                    json={"prompt": test["prompt"], "stream": False},
                )

                if response.status_code != 200:
                    print(f"❌ API Error: {response.status_code}")
                    print(f"   Response: {response.text[:200]}")
                    continue

                result = response.json()
                answer = result.get("response", "")

                # Check for anti-patterns (dummy code)
                found_antipatterns = []
                for pattern in ANTI_PATTERNS:
                    if pattern.lower() in answer.lower():
                        found_antipatterns.append(pattern)

                # Check for expected keywords (real implementation)
                found_keywords = []
                for keyword in test.get("expect_keywords", []):
                    if keyword.lower() in answer.lower():
                        found_keywords.append(keyword)

                # Analysis
                print(f"\n   Response length: {len(answer)} characters")

                if found_antipatterns:
                    print(f"   ⚠️  FOUND DUMMY CODE PATTERNS:")
                    for ap in found_antipatterns:
                        print(f"      - {ap}")
                else:
                    print(f"   ✅ No dummy code patterns detected")

                if found_keywords:
                    print(
                        f"   ✅ Real implementation detected ({len(found_keywords)}/{len(test.get('expect_keywords', []))} keywords)"
                    )
                    print(f"      Found: {', '.join(found_keywords[:3])}...")
                else:
                    print(f"   ⚠️  Expected keywords not found")

                # Check for actual code blocks
                code_blocks = answer.count("```")
                if code_blocks >= 2:
                    print(f"   ✅ Contains {code_blocks//2} code block(s)")

                # Show snippet
                print(f"\n   First 300 chars of response:")
                print(f"   {answer[:300]}...")

                # Overall assessment
                if not found_antipatterns and (found_keywords or code_blocks >= 2):
                    print(f"\n   ✅ PASS: Production-ready code generated")
                else:
                    print(f"\n   ❌ FAIL: Appears to be placeholder/dummy code")

            except Exception as e:
                print(f"❌ Error: {e}")

            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(test_code_generation())
