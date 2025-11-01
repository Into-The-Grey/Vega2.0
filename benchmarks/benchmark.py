#!/usr/bin/env python3
"""
Vega2.0 Performance Benchmark
Measure actual performance metrics on your hardware
"""

import subprocess
import time
import statistics
from pathlib import Path


def benchmark(name: str, prompt: str, iterations: int = 5):
    """Run benchmark and collect stats"""
    print(f"\n📊 {name}")
    print("=" * 50)

    times = []
    failures = 0

    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...", end=" ", flush=True)
        start = time.time()

        try:
            result = subprocess.run(
                ["/usr/bin/python3", "main.py", "cli", "chat", prompt],
                capture_output=True,
                timeout=30,
                cwd=Path(__file__).parent,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                times.append(elapsed)
                print(f"✅ {elapsed:.2f}s")
            else:
                failures += 1
                print(f"❌ Failed")
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"❌ Timeout")

    if times:
        avg = statistics.mean(times)
        median = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0

        print(f"\n  Average: {avg:.2f}s")
        print(f"  Median:  {median:.2f}s")
        print(f"  StdDev:  {stdev:.2f}s")
        print(f"  Min:     {min(times):.2f}s")
        print(f"  Max:     {max(times):.2f}s")
        if failures:
            print(f"  ⚠️  Failures: {failures}/{iterations}")

        return {
            "name": name,
            "avg": avg,
            "median": median,
            "stdev": stdev,
            "min": min(times),
            "max": max(times),
            "failures": failures,
            "total": iterations,
        }
    else:
        print(f"  ❌ All {iterations} runs failed")
        return None


def main():
    print("=" * 60)
    print("⚡ VEGA 2.0 PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("Measuring baseline performance on your hardware...\n")

    results = []

    # Benchmark 1: Simple response
    r1 = benchmark("Simple Response (5 iterations)", "What is 2+2?", iterations=5)
    if r1:
        results.append(r1)

    time.sleep(1)

    # Benchmark 2: Medium complexity
    r2 = benchmark(
        "Medium Complexity (5 iterations)",
        "Explain what Python is in 2 sentences",
        iterations=5,
    )
    if r2:
        results.append(r2)

    time.sleep(1)

    # Benchmark 3: Code generation
    r3 = benchmark(
        "Code Generation (3 iterations)",
        "Write a Python function that sorts a list",
        iterations=3,
    )
    if r3:
        results.append(r3)

    # Print summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"\n{r['name']}")
        print(f"  Average: {r['avg']:.2f}s")
        print(f"  Reliability: {r['total'] - r['failures']}/{r['total']} success")

        # Performance rating
        if r["avg"] < 2:
            print(f"  Rating: 🚀 Excellent")
        elif r["avg"] < 5:
            print(f"  Rating: ✅ Good")
        elif r["avg"] < 10:
            print(f"  Rating: ⚠️  Acceptable")
        else:
            print(f"  Rating: 🐌 Slow")

    print("\n" + "=" * 60)
    print("💡 OPTIMIZATION TIPS")
    print("=" * 60)

    overall_avg = statistics.mean([r["avg"] for r in results])

    if overall_avg < 5:
        print("\n✨ Your system is performing well!")
        print("   No immediate optimizations needed.")
    elif overall_avg < 10:
        print("\n⚡ Consider these optimizations:")
        print("   • Reduce GEN_TEMPERATURE to 0.5 (faster sampling)")
        print("   • Set GEN_TOP_K=10 (less computation)")
        print("   • Use a smaller model if available")
    else:
        print("\n🔧 Performance needs improvement:")
        print("   • Check system resources (CPU/RAM usage)")
        print("   • Close other applications")
        print("   • Consider using a smaller model")
        print("   • Reduce CONTEXT_WINDOW_SIZE in .env")


if __name__ == "__main__":
    main()
