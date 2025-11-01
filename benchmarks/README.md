# Benchmarks

This directory contains benchmark scripts and performance testing tools for Vega.

## Contents

- `benchmark.py` - Main benchmark script for performance testing

## Running Benchmarks

```bash
cd /home/ncacord/Vega2.0
python3 benchmarks/benchmark.py
```

## Benchmark Metrics

Benchmarks typically measure:

- Response time (latency)
- Throughput (requests per second)
- Memory usage
- CPU utilization
- Concurrent request handling

## Results

Benchmark results are saved to `/test_results/` directory with timestamps.

## Configuration

Benchmark parameters can be configured in the script or via command-line arguments. Check the script for available options.
