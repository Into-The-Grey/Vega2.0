#!/usr/bin/env python3
"""
Test Mistral model and monitor resource usage
Shows CPU, RAM, VRAM usage during inference
"""

import psutil
import time
import subprocess
import json
from datetime import datetime


def get_system_info():
    """Get total system resources"""
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)

    # RAM info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)

    # GPU info (try to get from nvidia-smi)
    gpu_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_info = []
            for line in lines:
                parts = line.split(",")
                if len(parts) == 2:
                    gpu_name = parts[0].strip()
                    gpu_mem = parts[1].strip()
                    gpu_info.append({"name": gpu_name, "memory": gpu_mem})
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return {
        "cpu_physical": cpu_count,
        "cpu_logical": cpu_count_logical,
        "ram_total_gb": ram_total_gb,
        "gpu": gpu_info,
    }


def get_current_usage():
    """Get current resource usage"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    # RAM usage
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_percent = ram.percent

    # GPU usage
    gpu_usage = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_usage = []
            for line in lines:
                parts = line.split(",")
                if len(parts) == 3:
                    gpu_util = float(parts[0].strip())
                    gpu_mem_used = float(parts[1].strip())
                    gpu_mem_total = float(parts[2].strip())
                    gpu_usage.append(
                        {
                            "utilization": gpu_util,
                            "memory_used_mb": gpu_mem_used,
                            "memory_total_mb": gpu_mem_total,
                            "memory_percent": (
                                (gpu_mem_used / gpu_mem_total * 100)
                                if gpu_mem_total > 0
                                else 0
                            ),
                        }
                    )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used_gb,
        "ram_percent": ram_percent,
        "gpu": gpu_usage,
    }


def test_ollama_model():
    """Test Mistral model via Ollama"""
    print("🧪 Testing Mistral model inference...")

    try:
        # Simple test prompt
        test_prompt = "What is the capital of France? Answer in one word."

        start_time = time.time()

        result = subprocess.run(
            ["ollama", "run", "mistral", test_prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )

        end_time = time.time()
        inference_time = end_time - start_time

        if result.returncode == 0:
            response = result.stdout.strip()
            return {
                "success": True,
                "prompt": test_prompt,
                "response": response,
                "inference_time": inference_time,
            }
        else:
            return {"success": False, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 30 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 80)
    print("VEGA Mistral Model Resource Usage Test")
    print("=" * 80)
    print()

    # Get system info
    print("📊 System Resources")
    print("-" * 80)
    sys_info = get_system_info()
    print(
        f"CPU Cores: {sys_info['cpu_physical']} physical, {sys_info['cpu_logical']} logical"
    )
    print(f"RAM Total: {sys_info['ram_total_gb']:.2f} GB")

    if sys_info["gpu"]:
        print("\nGPU(s) Detected:")
        for i, gpu in enumerate(sys_info["gpu"]):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory']})")
    else:
        print("GPU: None detected (CPU-only inference)")

    print()

    # Baseline usage (before model load)
    print("📈 Baseline Resource Usage (before model)")
    print("-" * 80)
    baseline = get_current_usage()
    print(f"CPU: {baseline['cpu_percent']:.1f}%")
    print(
        f"RAM: {baseline['ram_used_gb']:.2f} GB / {sys_info['ram_total_gb']:.2f} GB ({baseline['ram_percent']:.1f}%)"
    )

    if baseline["gpu"]:
        print("\nGPU:")
        for i, gpu in enumerate(baseline["gpu"]):
            print(
                f"  GPU {i}: {gpu['utilization']:.1f}% util, "
                f"{gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB "
                f"({gpu['memory_percent']:.1f}%)"
            )

    print()

    # Test model
    print("🚀 Running Model Inference...")
    print("-" * 80)

    # Get usage during inference
    result = test_ollama_model()

    if result["success"]:
        print(f"✅ Model responded in {result['inference_time']:.2f} seconds")
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response'][:200]}")  # First 200 chars
        print()
    else:
        print(f"❌ Model test failed: {result.get('error', 'Unknown error')}")
        print()

    # Get usage after inference
    print("📊 Resource Usage During/After Inference")
    print("-" * 80)

    # Check Ollama process
    ollama_processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        try:
            if "ollama" in proc.info["name"].lower():
                ollama_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if ollama_processes:
        print("Ollama Process(es):")
        for proc in ollama_processes:
            try:
                mem_mb = proc.info["memory_info"].rss / (1024**2)
                cpu_pct = proc.cpu_percent(interval=0.1)
                print(
                    f"  PID {proc.info['pid']}: {mem_mb:.1f} MB RAM, {cpu_pct:.1f}% CPU"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print()

    # Overall system usage
    current = get_current_usage()
    print(f"System CPU: {current['cpu_percent']:.1f}%")
    print(
        f"System RAM: {current['ram_used_gb']:.2f} GB / {sys_info['ram_total_gb']:.2f} GB "
        f"({current['ram_percent']:.1f}%)"
    )

    # Calculate delta from baseline
    cpu_delta = current["cpu_percent"] - baseline["cpu_percent"]
    ram_delta = current["ram_used_gb"] - baseline["ram_used_gb"]

    print(f"\nChange from baseline:")
    print(f"  CPU: {cpu_delta:+.1f}%")
    print(f"  RAM: {ram_delta:+.2f} GB")

    if current["gpu"]:
        print("\nGPU:")
        for i, gpu in enumerate(current["gpu"]):
            print(
                f"  GPU {i}: {gpu['utilization']:.1f}% util, "
                f"{gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB "
                f"({gpu['memory_percent']:.1f}%)"
            )

            if baseline["gpu"] and i < len(baseline["gpu"]):
                gpu_mem_delta = (
                    gpu["memory_used_mb"] - baseline["gpu"][i]["memory_used_mb"]
                )
                print(f"    Change: {gpu_mem_delta:+.0f} MB")

    print()
    print("=" * 80)
    print("💡 Notes:")
    print("  - Mistral 7B Q4_K_M quantization: ~4.4 GB on disk")
    print("  - First inference may be slower (model loading)")
    print("  - CPU inference typical: 5-30 tokens/sec (depends on CPU)")
    print("  - GPU inference: 50-200+ tokens/sec (if CUDA available)")
    print("=" * 80)


if __name__ == "__main__":
    main()
