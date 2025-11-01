# Test Results

This directory contains results from various test runs.

## File Types

### JSON Results

- `censorship_test_*.json` - Ethics and censorship test results with timestamps
- `topic_test_*.json` - Topic range test results
- `stress_test_results.json` - Performance test metrics

### Log Files

- `benchmark_results.log` - Benchmark execution logs
- `extreme_test_results.log` - Extreme stress test logs
- `topic_test_output.txt` - Topic test console output

## Reading Results

Most results are in JSON format with the following structure:

```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "test_name": "test_category",
  "results": [...],
  "summary": {...}
}
```

## Cleaning Up

To clean old test results:

```bash
# Remove results older than 7 days
find test_results/ -name "*.json" -mtime +7 -delete
find test_results/ -name "*.log" -mtime +7 -delete
```

## Latest Test Results

Check the most recent files by modification time:

```bash
ls -lt test_results/ | head -10
```
