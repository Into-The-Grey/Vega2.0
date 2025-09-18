#!/usr/bin/env python3
"""
Start the Vega2.0 OpenAPI server for testing
"""

import uvicorn
import sys
import os

# Add the project root and src to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def main():
    """Main function to start the OpenAPI server."""
    print("Starting Vega2.0 OpenAPI Server...")
    print("Documentation will be available at:")
    print("  - Swagger UI: http://127.0.0.1:8001/docs")
    print("  - ReDoc: http://127.0.0.1:8001/redoc")
    print("  - OpenAPI JSON: http://127.0.0.1:8001/openapi.json")

    try:
        from src.vega.core.openapi_app import app
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            reload=False
        )
    except ImportError as e:
        print(f"❌ Failed to import OpenAPI app: {e}")
        print("Make sure the core modules are available in src/vega/core/")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
