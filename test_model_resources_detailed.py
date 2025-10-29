#!/usr/bin/env python3
"""
Detailed Mistral Model Resource Report
Shows comprehensive system resource usage and model statistics
"""

import psutil
import subprocess
import json
from pathlib import Path


def format_bytes(bytes_val):
    """Format bytes to human-readable"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_ollama_model_info():
    """Get Ollama model information"""
    try:
        result = subprocess.run(
            ["ollama", "show", "mistral", "--modelfile"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return None


def get_disk_usage():
    """Get disk usage for Ollama models"""
    try:
        # Ollama models typically stored in ~/.ollama/models
        ollama_dir = Path.home() / ".ollama" / "models"
        if ollama_dir.exists():
            total_size = 0
            model_files = []
            for file in ollama_dir.rglob("*"):
                if file.is_file():
                    size = file.stat().st_size
                    total_size += size
                    if size > 100_000_000:  # Files > 100MB
                        model_files.append(
                            {"path": str(file.relative_to(ollama_dir)), "size": size}
                        )
            return {
                "total_size": total_size,
                "large_files": sorted(
                    model_files, key=lambda x: x["size"], reverse=True
                )[:5],
            }
    except:
        pass
    return None


def main():
    print("=" * 80)
    print("🔍 VEGA Mistral Model - Detailed Resource Report")
    print("=" * 80)
    print()

    # System Overview
    print("🖥️  SYSTEM OVERVIEW")
    print("-" * 80)

    # CPU
    cpu_freq = psutil.cpu_freq()
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)

    print(f"CPU:")
    print(f"  Physical cores: {cpu_count_physical}")
    print(f"  Logical cores: {cpu_count_logical}")
    if cpu_freq:
        print(f"  Current freq: {cpu_freq.current:.0f} MHz")
        print(f"  Max freq: {cpu_freq.max:.0f} MHz")

    # RAM
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"\nRAM:")
    print(f"  Total: {format_bytes(ram.total)}")
    print(f"  Available: {format_bytes(ram.available)} ({100 - ram.percent:.1f}% free)")
    print(f"  Used: {format_bytes(ram.used)} ({ram.percent:.1f}%)")
    print(f"  Buffers/Cache: {format_bytes(ram.buffers + ram.cached)}")

    print(f"\nSwap:")
    print(f"  Total: {format_bytes(swap.total)}")
    print(f"  Used: {format_bytes(swap.used)} ({swap.percent:.1f}%)")
    print(f"  Free: {format_bytes(swap.free)}")

    # GPU
    print(f"\nGPU:")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    idx, name, driver, total, used, free = parts[:6]
                    temp = parts[6] if len(parts) > 6 else "N/A"
                    power_draw = parts[7] if len(parts) > 7 else "N/A"
                    power_limit = parts[8] if len(parts) > 8 else "N/A"

                    print(f"  GPU {idx}: {name}")
                    print(f"    Driver: {driver}")
                    print(f"    Memory: {used} / {total} ({free} free)")
                    print(f"    Temperature: {temp}")
                    print(f"    Power: {power_draw} / {power_limit}")
        else:
            print("  No NVIDIA GPU detected")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  nvidia-smi not available (CPU-only mode)")

    # Disk usage
    disk = psutil.disk_usage("/")
    print(f"\nDisk (root):")
    print(f"  Total: {format_bytes(disk.total)}")
    print(f"  Used: {format_bytes(disk.used)} ({disk.percent:.1f}%)")
    print(f"  Free: {format_bytes(disk.free)}")

    print()

    # Ollama/Model Info
    print("🤖 MODEL INFORMATION")
    print("-" * 80)

    # Check if Ollama is running
    ollama_running = False
    ollama_procs = []
    for proc in psutil.process_iter(
        ["pid", "name", "memory_info", "cpu_percent", "create_time"]
    ):
        try:
            if "ollama" in proc.info["name"].lower():
                ollama_running = True
                ollama_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if ollama_running:
        print("Ollama Status: ✅ Running")
        print("\nOllama Processes:")
        for proc in ollama_procs:
            try:
                mem_mb = proc.info["memory_info"].rss / (1024**2)
                cpu_pct = proc.cpu_percent(interval=0.1)
                print(
                    f"  PID {proc.info['pid']}: {mem_mb:.1f} MB RAM, {cpu_pct:.1f}% CPU"
                )
            except:
                pass
    else:
        print("Ollama Status: ❌ Not running")

    print()

    # Model file size
    disk_info = get_disk_usage()
    if disk_info:
        print(f"Ollama Models Directory:")
        print(f"  Total size: {format_bytes(disk_info['total_size'])}")
        if disk_info["large_files"]:
            print(f"  Large model files:")
            for file_info in disk_info["large_files"]:
                print(f"    {format_bytes(file_info['size'])}: {file_info['path']}")

    print()

    # Mistral model info
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("Installed Models:")
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                if "mistral" in line.lower():
                    print(f"  ✅ {line}")
    except:
        pass

    # Model details
    model_info = get_ollama_model_info()
    if model_info:
        print("\nMistral Model Configuration:")
        for line in model_info.split("\n"):
            if line.strip():
                print(f"  {line}")

    print()

    # Resource Requirements
    print("📊 MISTRAL 7B RESOURCE REQUIREMENTS")
    print("-" * 80)
    print("Quantization: Q4_K_M (4-bit quantized)")
    print()
    print("Disk Space:")
    print("  Model file: ~4.4 GB")
    print("  With cache: ~5-6 GB")
    print()
    print("RAM (CPU Inference):")
    print("  Minimum: 8 GB")
    print("  Recommended: 16 GB")
    print("  Model loaded: ~4.5-5 GB")
    print("  Inference overhead: ~1-2 GB")
    print("  Total typical: ~6-8 GB")
    print()
    print("VRAM (GPU Inference):")
    print("  Model on GPU: ~5-6 GB")
    print("  Minimum GPU: 6 GB VRAM")
    print("  Recommended: 8+ GB VRAM")
    print()
    print("CPU:")
    print("  Minimum: 4 cores")
    print("  Recommended: 8+ cores")
    print("  Usage during inference: 50-100% (depends on threads)")
    print()

    # Your System Capacity
    print("✅ YOUR SYSTEM CAPACITY")
    print("-" * 80)

    ram_available_gb = ram.available / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    print(f"RAM Available: {ram_available_gb:.1f} GB / {ram_total_gb:.1f} GB")
    if ram_available_gb >= 8:
        print("  ✅ Sufficient for CPU inference")
    else:
        print("  ⚠️  Low available RAM (may impact performance)")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            gpus = [float(x.strip()) for x in result.stdout.strip().split("\n")]
            print(f"\nGPU VRAM Available:")
            for i, vram_free in enumerate(gpus):
                vram_free_gb = vram_free / 1024
                print(f"  GPU {i}: {vram_free_gb:.1f} GB free")
                if vram_free_gb >= 6:
                    print(f"    ✅ Sufficient for GPU inference")
                elif vram_free_gb >= 4:
                    print(f"    ⚠️  Tight fit (may work with optimizations)")
                else:
                    print(f"    ❌ Insufficient for full model (use CPU)")
    except:
        pass

    print(f"\nCPU: {cpu_count_physical} physical cores")
    if cpu_count_physical >= 8:
        print("  ✅ Excellent for fast inference")
    elif cpu_count_physical >= 4:
        print("  ✅ Sufficient for inference")
    else:
        print("  ⚠️  Minimum requirements met")

    print()

    # Performance Estimates
    print("⚡ ESTIMATED PERFORMANCE")
    print("-" * 80)
    print("Based on your system:")
    print()

    # Check which GPU is being used
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = line.split(",")
                if len(parts) == 2:
                    gpu_name = parts[0].strip()
                    vram_free = float(parts[1].strip().split()[0])

                    if vram_free >= 6000:  # 6GB+ available
                        print(f"GPU {i} ({gpu_name}):")

                        if "1660" in gpu_name or "1650" in gpu_name:
                            print("  Tokens/sec: ~40-60")
                            print("  Small response (50 tokens): ~1-2 seconds")
                            print("  Medium response (200 tokens): ~4-6 seconds")
                        elif (
                            "RTX" in gpu_name
                            or "3060" in gpu_name
                            or "3070" in gpu_name
                        ):
                            print("  Tokens/sec: ~80-120")
                            print("  Small response (50 tokens): ~0.5-1 second")
                            print("  Medium response (200 tokens): ~2-3 seconds")
                        else:
                            print("  Tokens/sec: ~30-50 (varies by GPU)")
                            print("  Small response (50 tokens): ~1-2 seconds")
                            print("  Medium response (200 tokens): ~4-8 seconds")
                        print()
    except:
        pass

    print("CPU Inference (fallback):")
    print("  Tokens/sec: ~5-15")
    print("  Small response (50 tokens): ~3-10 seconds")
    print("  Medium response (200 tokens): ~15-40 seconds")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
