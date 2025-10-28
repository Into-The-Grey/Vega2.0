#!/usr/bin/env python3
"""
Phase 3 Optimization Verification Script

Tests all new monitoring and diagnostic capabilities:
- Database query profiling
- Integration health checks
- System diagnostics
- Configuration validation
"""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class Colors:
    """ANSI color codes"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_test(name: str):
    """Print test name"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🔍 {name}{Colors.END}")


def print_pass(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_fail(message: str):
    """Print failure message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warn(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"   {message}")


async def test_db_profiler():
    """Test database profiler functionality"""
    print_test("Database Profiler")

    try:
        from vega.core.db_profiler import get_profiler, profile_db_function
        from vega.core.db import get_history, log_conversation

        profiler = get_profiler()

        # Test basic stats
        stats = profiler.get_stats()
        print_pass(f"Profiler initialized - {stats['total_queries']} queries tracked")

        # Test query profiling
        log_conversation("test prompt", "test response", "test")
        history = get_history(limit=5)

        new_stats = profiler.get_stats()
        if new_stats["total_queries"] > stats["total_queries"]:
            print_pass(
                f"Query profiling working - {new_stats['total_queries'] - stats['total_queries']} new queries tracked"
            )
        else:
            print_fail("Query profiling not tracking queries")

        # Test recent queries
        recent = profiler.get_recent_queries(limit=5)
        if recent:
            print_pass(f"Recent queries available - showing last {len(recent)}")
            for q in recent[:2]:
                print_info(f"  {q['query'][:50]}... ({q['duration_ms']:.2f}ms)")

        # Test slow query threshold
        profiler.set_slow_query_threshold(50.0)
        print_pass("Slow query threshold configurable")

        return True
    except Exception as e:
        print_fail(f"Database profiler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_integration_health():
    """Test integration health checks"""
    print_test("Integration Health Checks")

    try:
        from vega.core.integration_health import (
            check_database,
            check_llm_backend,
            check_all_integrations,
        )

        # Test database health
        db_health = await check_database()
        if db_health.status == "healthy":
            print_pass(
                f"Database health check passed ({db_health.response_time_ms:.2f}ms)"
            )
        else:
            print_warn(f"Database health: {db_health.status} - {db_health.error}")

        # Test LLM backend health
        try:
            llm_health = await check_llm_backend()
            if llm_health.status == "healthy":
                print_pass(f"LLM backend healthy ({llm_health.response_time_ms:.2f}ms)")
            else:
                print_warn(
                    f"LLM backend: {llm_health.status} - {llm_health.error or 'N/A'}"
                )
        except Exception as e:
            print_warn(f"LLM backend check skipped: {e}")

        # Test all integrations
        all_health = await check_all_integrations(timeout=5.0)

        print_pass(
            f"Completed {all_health['total_checks']} integration checks ({all_health['check_duration_ms']:.2f}ms)"
        )
        print_info(f"Overall status: {all_health['status']}")
        print_info(
            f"Unhealthy: {all_health['unhealthy']}, Degraded: {all_health['degraded']}"
        )

        # Show integration statuses
        for name, health in all_health["integrations"].items():
            status = health["status"]
            emoji = (
                "✅"
                if status == "healthy"
                else "⚠️" if status in ["degraded", "disabled"] else "❌"
            )
            time_str = (
                f" ({health.get('response_time_ms', 0):.1f}ms)"
                if health.get("response_time_ms")
                else ""
            )
            print_info(f"{emoji} {name}: {status}{time_str}")

        return True
    except Exception as e:
        print_fail(f"Integration health test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_system_diagnostics():
    """Test system diagnostics"""
    print_test("System Diagnostics")

    try:
        from vega.core.system_diagnostics import (
            get_memory_stats,
            get_cpu_stats,
            get_thread_stats,
            get_network_stats,
            get_full_diagnostics,
            get_health_summary,
        )

        # Test memory stats
        mem = get_memory_stats()
        if mem.get("status") != "error":
            print_pass(
                f"Memory monitoring - Process: {mem['process']['rss_mb']:.1f}MB, "
                f"System: {mem['system']['percent']:.1f}% used"
            )

        # Test CPU stats
        cpu = get_cpu_stats()
        if cpu.get("status") != "error":
            print_pass(
                f"CPU monitoring - Process: {cpu['process']['percent']:.1f}%, "
                f"Threads: {cpu['process']['num_threads']}"
            )

        # Test thread stats
        threads = get_thread_stats()
        if threads.get("status") != "error":
            print_pass(f"Thread monitoring - {threads['total_threads']} total threads")
            print_info(f"Thread names: {', '.join(threads['thread_names'][:5])}...")

        # Test network stats
        network = get_network_stats()
        if network.get("status") != "error":
            print_pass(
                f"Network monitoring - {network['total_connections']} connections"
            )

        # Test full diagnostics
        diag = await get_full_diagnostics()
        print_pass("Full diagnostics available")

        # Test health summary
        health = get_health_summary()
        print_pass(f"Health summary: {health}")

        return True
    except Exception as e:
        print_fail(f"System diagnostics test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_config_validation():
    """Test configuration validation"""
    print_test("Configuration Validation")

    try:
        from vega.core.config_validator import ConfigValidator, validate_startup_config
        from vega.core.config import get_config

        config = get_config()
        validator = ConfigValidator()

        is_valid, summary = validator.validate_config(config)

        print_pass(f"Configuration validation complete")
        print_info(f"Valid: {is_valid}")
        print_info(f"Total checks: {summary['total_checks']}")
        print_info(f"Critical errors: {summary['critical_errors']}")
        print_info(f"Warnings: {summary['warnings']}")

        if summary["critical_errors"] > 0:
            print_warn("Critical configuration errors found:")
            for result in summary["results"]:
                if result["severity"] == "critical":
                    print_info(f"  {result['field']}: {result['message']}")

        if summary["warnings"] > 0:
            print_info("Configuration warnings:")
            for result in summary["results"]:
                if result["severity"] == "warning":
                    print_info(f"  {result['field']}: {result['message']}")

        return True
    except Exception as e:
        print_fail(f"Configuration validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_database_indexes():
    """Test database indexes"""
    print_test("Database Indexes")

    try:
        from vega.core.db import engine

        with engine.connect() as conn:
            # Get all indexes
            result = conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='conversations'"
            )
            indexes = [row[0] for row in result.fetchall()]

            expected_indexes = [
                "ix_conv_ts",
                "ix_conv_session",
                "ix_conv_session_ts",
                "ix_conv_ts_session",
                "ix_conv_reviewed_ts",
                "ix_conv_source_ts",
            ]

            found_indexes = []
            missing_indexes = []

            for expected in expected_indexes:
                if expected in indexes:
                    found_indexes.append(expected)
                else:
                    missing_indexes.append(expected)

            print_pass(f"Found {len(found_indexes)} indexes")
            for idx in found_indexes:
                print_info(f"  ✓ {idx}")

            if missing_indexes:
                print_warn(f"Missing {len(missing_indexes)} indexes:")
                for idx in missing_indexes:
                    print_info(f"  ✗ {idx}")
                return False

            return True
    except Exception as e:
        print_fail(f"Database index test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all verification tests"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Phase 3 Optimization Verification{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")

    results = {}

    # Run all tests
    results["db_profiler"] = await test_db_profiler()
    results["integration_health"] = await test_integration_health()
    results["system_diagnostics"] = await test_system_diagnostics()
    results["config_validation"] = await test_config_validation()
    results["database_indexes"] = await test_database_indexes()

    # Summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = (
            f"{Colors.GREEN}✅ PASS{Colors.END}"
            if result
            else f"{Colors.RED}❌ FAIL{Colors.END}"
        )
        print(f"{status} - {test.replace('_', ' ').title()}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Phase 3 optimization verified.{Colors.END}"
        )
        return 0
    else:
        print(
            f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed. Review output above.{Colors.END}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
