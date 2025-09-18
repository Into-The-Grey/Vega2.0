#!/usr/bin/env python3
"""
Vega2.0 - Main Application Entry Point

This is the main entry point for the Vega2.0 autonomous AI system.
All components have been organized into a clean modular structure.

Directory Structure:
- core/: Core application components (app.py, cli.py, db.py, etc.)
- sac/: System Autonomy Core modules
- intelligence/: AI intelligence and analysis engines
- analysis/: Conversation and data analysis tools
- ui/: User interface components and static files
- data/: Databases, configurations, and data files
- integrations/: External service integrations
- datasets/: Dataset preparation and training data
- training/: Model training and fine-tuning
- learning/: Learning and evaluation systems
- docs/: Documentation and guides
- book/: mdBook documentation
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent / "sac"))
sys.path.insert(0, str(Path(__file__).parent / "intelligence"))


def main():
    """Main application entry point"""
    print("🚀 Vega2.0 - Autonomous AI System")
    print("📁 Project structure has been reorganized for better maintainability")
    print()
    print("Available components:")
    print("  🧠 Core System:          python -m core.app")
    print("  💬 CLI Interface:        python -m core.cli")
    print("  🤖 System Autonomy:      python -m sac.self_govern")
    print("  🎛️  System Interface:     python -m sac.system_interface")
    print("  🔍 System Probe:         python -m sac.system_probe")
    print("  👁️  System Watchdog:      python -m sac.system_watchdog")
    print("  ⚙️  System Control:       python -m sac.sys_control")
    print("  🛡️  Network Guard:        python -m sac.net_guard")
    print("  💰 Economic Scanner:     python -m sac.economic_scanner")
    print("  📊 Dashboard:            python -m ui.dashboard")
    print()
    print("For detailed usage, see docs/ or run specific modules with --help")


if __name__ == "__main__":
    main()
