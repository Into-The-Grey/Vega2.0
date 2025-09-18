#!/usr/bin/env python3
"""
Start the Vega2.0 OpenAPI server for testing
"""

import uvicorn
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("Starting Vega2.0 OpenAPI Server...")
    print("Documentation will be available at:")
    print("  - Swagger UI: http://127.0.0.1:8001/docs")
    print("  - ReDoc: http://127.0.0.1:8001/redoc")
    print("  - OpenAPI JSON: http://127.0.0.1:8001/openapi.json")

    uvicorn.run(
        "core.openapi_app:app",
        host="127.0.0.1",
        port=8001,  # Use different port from main app
        reload=True,
        log_level="info",
    )
