#!/usr/bin/env python3
"""
Vega2.0 Main Entry Point
========================

This script provides the main entry point for running Vega2.0 applications
with the new organized structure.

Usage:
    python main.py [command] [options]
    
Commands:
    server      Start the main API server
    cli         Run CLI interface
    openapi     Start OpenAPI server
    processes   Start background processes
    test        Run test suite
    
Examples:
    python main.py server --host 127.0.0.1 --port 8000
    python main.py cli chat "Hello!"
    python main.py openapi --port 8001
    python main.py processes start
    python main.py test --suite core
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main entry point for Vega2.0."""
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    try:
        if command == "server":
            from src.vega.core.app import app
            import uvicorn
            
            host = "127.0.0.1"
            port = 8000
            
            # Parse basic args
            if "--host" in args:
                host = args[args.index("--host") + 1]
            if "--port" in args:
                port = int(args[args.index("--port") + 1])
                
            print(f"🚀 Starting Vega2.0 API server on {host}:{port}")
            uvicorn.run(app, host=host, port=port)
            
        elif command == "cli":
            from src.vega.core.cli import main as cli_main
            # Pass remaining args to CLI
            sys.argv = ["cli"] + args
            cli_main()
            
        elif command == "openapi":
            from scripts.run_openapi_server import main as openapi_main
            openapi_main()
            
        elif command == "processes":
            from scripts.run_processes import main as processes_main
            processes_main()
            
        elif command == "test":
            import subprocess
            if args and args[0] == "--suite":
                suite = args[1] if len(args) > 1 else "all"
                subprocess.run([sys.executable, "-m", "pytest", f"tests/test_{suite}.py", "-v"])
            else:
                subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
                
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()