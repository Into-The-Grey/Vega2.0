"""
Test Suite FastAPI Application for Vega2.0
===========================================

Dedicated test runner with UI for comprehensive testing of all Vega2.0 components.
Includes dummy parameters and test configurations isolated from production.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_CONFIG = {
    "dummy_api_keys": {
        "GOOGLE_CALENDAR_CREDENTIALS": "test_credentials.json",
        "PLAID_CLIENT_ID": "test_plaid_client",
        "PLAID_SECRET": "test_plaid_secret",
        "OPENAI_API_KEY": "test_openai_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
    },
    "test_data_paths": {
        "syllabus_directory": "test_suite/test_data/education",
        "financial_csv_directory": "test_suite/test_data/financial",
        "voice_test_files": "test_suite/test_data/voice",
    },
    "mock_endpoints": {
        "ollama_url": "http://localhost:11434",
        "test_calendar_api": "https://test-calendar-api.example.com",
    },
}

# Initialize FastAPI app
app = FastAPI(
    title="Vega2.0 Test Suite",
    description="Comprehensive testing interface for all Vega2.0 components",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Create test data directories
for path in TEST_CONFIG["test_data_paths"].values():
    os.makedirs(path, exist_ok=True)

# Logging setup for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_suite/test_results.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Test result storage
test_results: Dict[str, Any] = {}


class TestRequest(BaseModel):
    test_category: str
    test_name: str
    parameters: Optional[Dict[str, Any]] = None


class TestResult(BaseModel):
    test_id: str
    status: str
    duration: float
    details: Dict[str, Any]
    timestamp: datetime


@app.get("/", response_class=HTMLResponse)
async def test_suite_home():
    """Main test suite interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vega2.0 Test Suite</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .test-category { border: 1px solid #ccc; margin: 10px 0; padding: 15px; }
            .test-button { background: #007bff; color: white; padding: 10px 15px; border: none; margin: 5px; cursor: pointer; }
            .test-button:hover { background: #0056b3; }
            .results { background: #f8f9fa; padding: 15px; margin: 10px 0; }
            .pass { color: green; }
            .fail { color: red; }
            .running { color: orange; }
        </style>
    </head>
    <body>
        <h1>🧪 Vega2.0 Test Suite</h1>
        
        <div class="test-category">
            <h2>🔌 Integration Tests</h2>
            <button class="test-button" onclick="runTest('integration', 'calendar_sync')">Calendar Sync</button>
            <button class="test-button" onclick="runTest('integration', 'finance_monitor')">Finance Monitor</button>
            <button class="test-button" onclick="runTest('integration', 'llm_providers')">LLM Providers</button>
            <button class="test-button" onclick="runTest('integration', 'voice_processing')">Voice Processing</button>
        </div>
        
        <div class="test-category">
            <h2>🧠 Core Component Tests</h2>
            <button class="test-button" onclick="runTest('core', 'config_loading')">Config Loading</button>
            <button class="test-button" onclick="runTest('core', 'database_operations')">Database Operations</button>
            <button class="test-button" onclick="runTest('core', 'error_handling')">Error Handling</button>
            <button class="test-button" onclick="runTest('core', 'logging_system')">Logging System</button>
        </div>
        
        <div class="test-category">
            <h2>🎯 Performance Tests</h2>
            <button class="test-button" onclick="runTest('performance', 'response_time')">Response Time</button>
            <button class="test-button" onclick="runTest('performance', 'memory_usage')">Memory Usage</button>
            <button class="test-button" onclick="runTest('performance', 'concurrent_requests')">Concurrent Requests</button>
        </div>
        
        <div class="test-category">
            <h2>🔄 Background Process Tests</h2>
            <button class="test-button" onclick="runTest('background', 'system_monitoring')">System Monitoring</button>
            <button class="test-button" onclick="runTest('background', 'data_sync')">Data Sync</button>
            <button class="test-button" onclick="runTest('background', 'cleanup_tasks')">Cleanup Tasks</button>
        </div>
        
        <button class="test-button" onclick="runAllTests()" style="background: #28a745; font-size: 18px; padding: 15px 30px;">
            🚀 Run All Tests
        </button>
        
        <div id="results" class="results">
            <h3>📊 Test Results</h3>
            <div id="test-output">No tests run yet</div>
        </div>
        
        <script>
            async function runTest(category, testName) {
                const output = document.getElementById('test-output');
                output.innerHTML += `<div class="running">🔄 Running ${category}/${testName}...</div>`;
                
                try {
                    const response = await fetch('/run_test', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            test_category: category,
                            test_name: testName
                        })
                    });
                    
                    const result = await response.json();
                    const statusClass = result.status === 'PASS' ? 'pass' : 'fail';
                    const statusIcon = result.status === 'PASS' ? '✅' : '❌';
                    
                    output.innerHTML += `<div class="${statusClass}">${statusIcon} ${category}/${testName}: ${result.status} (${result.duration.toFixed(2)}s)</div>`;
                    
                    if (result.details.error) {
                        output.innerHTML += `<div style="margin-left: 20px; color: #666;">Error: ${result.details.error}</div>`;
                    }
                } catch (error) {
                    output.innerHTML += `<div class="fail">❌ ${category}/${testName}: ERROR - ${error}</div>`;
                }
            }
            
            async function runAllTests() {
                document.getElementById('test-output').innerHTML = '<div class="running">🔄 Running full test suite...</div>';
                const response = await fetch('/run_all_tests', {method: 'POST'});
                const results = await response.json();
                
                let output = '<h4>📋 Full Test Suite Results</h4>';
                for (const result of results) {
                    const statusClass = result.status === 'PASS' ? 'pass' : 'fail';
                    const statusIcon = result.status === 'PASS' ? '✅' : '❌';
                    output += `<div class="${statusClass}">${statusIcon} ${result.test_id}: ${result.status} (${result.duration.toFixed(2)}s)</div>`;
                }
                
                document.getElementById('test-output').innerHTML = output;
            }
        </script>
    </body>
    </html>
    """


@app.post("/run_test")
async def run_single_test(test_request: TestRequest) -> TestResult:
    """Run a single test"""
    test_id = f"{test_request.test_category}.{test_request.test_name}"
    start_time = time.time()

    try:
        # Import test modules dynamically to avoid dependency issues
        test_module = __import__(
            f"test_suite.tests.test_{test_request.test_category}",
            fromlist=[test_request.test_name],
        )
        test_function = getattr(test_module, f"test_{test_request.test_name}")

        # Run the test with dummy parameters
        result = await test_function(TEST_CONFIG)

        duration = time.time() - start_time
        test_result = TestResult(
            test_id=test_id,
            status="PASS" if result.get("success", False) else "FAIL",
            duration=duration,
            details=result,
            timestamp=datetime.now(),
        )

        test_results[test_id] = test_result
        logger.info(f"Test {test_id} completed: {test_result.status}")

        return test_result

    except Exception as e:
        duration = time.time() - start_time
        test_result = TestResult(
            test_id=test_id,
            status="FAIL",
            duration=duration,
            details={"success": False, "error": str(e)},
            timestamp=datetime.now(),
        )

        test_results[test_id] = test_result
        logger.error(f"Test {test_id} failed: {e}")

        return test_result


@app.post("/run_all_tests")
async def run_all_tests() -> List[TestResult]:
    """Run all available tests"""
    all_tests = [
        ("integration", "calendar_sync"),
        ("integration", "finance_monitor"),
        ("integration", "llm_providers"),
        ("core", "config_loading"),
        ("core", "database_operations"),
        ("performance", "response_time"),
    ]

    results = []
    for category, test_name in all_tests:
        try:
            result = await run_single_test(
                TestRequest(test_category=category, test_name=test_name)
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to run test {category}.{test_name}: {e}")
            continue

    return results


@app.get("/test_results")
async def get_test_results() -> Dict[str, Any]:
    """Get all test results"""
    return {
        "results": test_results,
        "summary": {
            "total": len(test_results),
            "passed": sum(1 for r in test_results.values() if r.status == "PASS"),
            "failed": sum(1 for r in test_results.values() if r.status == "FAIL"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
