#!/usr/bin/env python3
"""
Import Path Fixer for Vega2.0 Reorganization
============================================

This script fixes all import paths to work with the new src/vega structure.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Map of old import patterns to new ones
        import_mappings = {
            # Core module imports
            r'\bfrom config import': 'from ..config import',
            r'\bfrom db import': 'from ..db import',
            r'\bfrom llm import': 'from ..llm import',
            r'\bfrom memory import': 'from ..memory import',
            r'\bfrom security import': 'from ..security import',
            r'\bfrom resilience import': 'from ..resilience import',
            
            # Relative imports within core
            r'\bfrom \.config import': 'from .config import',
            r'\bfrom \.db import': 'from .db import',
            r'\bfrom \.llm import': 'from .llm import',
            r'\bfrom \.memory import': 'from .memory import',
            
            # Cross-module imports
            r'\bfrom integrations\.': 'from ..integrations.',
            r'\bfrom voice\.': 'from ..voice.',
            r'\bfrom intelligence\.': 'from ..intelligence.',
            r'\bfrom datasets\.': 'from ..datasets.',
            r'\bfrom training\.': 'from ..training.',
            r'\bfrom learning\.': 'from ..learning.',
            r'\bfrom user\.': 'from ..user.',
            
            # Core imports from other files
            r'\bfrom core\.': 'from .',
            r'\bimport core\.': 'from . import',
            
            # Utils imports (now in tools)
            r'\bfrom utils\.': 'from ...tools.utils.',
            r'\bimport utils\.': 'import tools.utils.',
        }
        
        # Apply the mappings
        for old_pattern, new_pattern in import_mappings.items():
            content = re.sub(old_pattern, new_pattern, content)
        
        # Special handling for scripts that need to import from src
        if '/scripts/' in str(file_path):
            content = re.sub(r'\bfrom core\.', 'from src.vega.core.', content)
            content = re.sub(r'\bfrom src\.vega\.core\.\.\.(.*?)', r'from src.vega.\1', content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed imports in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False

def fix_all_imports():
    """Fix imports in all Python files."""
    project_root = Path(".").absolute()  # Use current working directory
    
    # Files to process
    python_files = []
    
    # Core modules
    core_dir = project_root / "src" / "vega" / "core"
    if core_dir.exists():
        python_files.extend(core_dir.glob("*.py"))
        print(f"Found {len(list(core_dir.glob('*.py')))} core files")
    
    # Other vega modules
    vega_dir = project_root / "src" / "vega"
    for subdir in ["integrations", "voice", "intelligence", "datasets", "training", "learning", "user"]:
        module_dir = vega_dir / subdir
        if module_dir.exists():
            files = list(module_dir.rglob("*.py"))
            python_files.extend(files)
            print(f"Found {len(files)} files in {subdir}")
    
    # Scripts
    scripts_dir = project_root / "scripts"
    if scripts_dir.exists():
        files = list(scripts_dir.glob("*.py"))
        python_files.extend(files)
        print(f"Found {len(files)} script files")
    
    # Tests
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        files = list(tests_dir.glob("*.py"))
        python_files.extend(files)
        print(f"Found {len(files)} test files")
    
    # Tools
    tools_dir = project_root / "tools"
    if tools_dir.exists():
        files = list(tools_dir.rglob("*.py"))
        python_files.extend(files)
        print(f"Found {len(files)} tool files")
    
    print(f"\nFound {len(python_files)} Python files to process...")
    
    fixed_count = 0
    for file_path in python_files:
        if file_path.name != "__pycache__" and file_path.is_file():
            if fix_imports_in_file(file_path):
                fixed_count += 1
    
    print(f"\n✓ Fixed imports in {fixed_count} files")

if __name__ == "__main__":
    fix_all_imports()