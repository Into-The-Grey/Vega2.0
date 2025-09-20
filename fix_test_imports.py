#!/usr/bin/env python3
"""
Script to fix import paths in test files
"""

import os
import re
import glob


def fix_imports_in_file(filepath):
    """Fix import statements in a single file"""
    print(f"Processing {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    original_content = content

    # Fix patterns for federated learning imports
    patterns = [
        # Direct module imports without src.vega prefix
        (
            r"from vega\.federated\.([a-zA-Z_]+) import",
            r"from src.vega.federated.\1 import",
        ),
        (r"import vega\.federated\.([a-zA-Z_]+)", r"import src.vega.federated.\1"),
        # Fallback imports that need src.vega prefix
        (r"from dp import", r"from src.vega.federated.dp import"),
        (r"from fedavg import", r"from src.vega.federated.fedavg import"),
        (
            r"from gradient_compression import",
            r"from src.vega.federated.gradient_compression import",
        ),
        (
            r"from homomorphic_encryption import",
            r"from src.vega.federated.homomorphic_encryption import",
        ),
        (r"from key_exchange import", r"from src.vega.federated.key_exchange import"),
        (r"from model_pruning import", r"from src.vega.federated.model_pruning import"),
        (r"from smpc import", r"from src.vega.federated.smpc import"),
        (
            r"from threshold_secret_sharing import",
            r"from src.vega.federated.threshold_secret_sharing import",
        ),
        # Import fallbacks without module prefix
        (r"import dp$", r"import src.vega.federated.dp as dp"),
        (r"import fedavg$", r"import src.vega.federated.fedavg as fedavg"),
        (
            r"import gradient_compression$",
            r"import src.vega.federated.gradient_compression as gradient_compression",
        ),
        (
            r"import homomorphic_encryption$",
            r"import src.vega.federated.homomorphic_encryption as homomorphic_encryption",
        ),
        (
            r"import key_exchange$",
            r"import src.vega.federated.key_exchange as key_exchange",
        ),
        (
            r"import model_pruning$",
            r"import src.vega.federated.model_pruning as model_pruning",
        ),
        (r"import smpc$", r"import src.vega.federated.smpc as smpc"),
        (
            r"import threshold_secret_sharing$",
            r"import src.vega.federated.threshold_secret_sharing as threshold_secret_sharing",
        ),
        # Fix tests relative imports
        (r"from \.([a-zA-Z_]+) import", r"from tests.\1 import"),
        (r"from \.\.([a-zA-Z_\.]+) import", r"from src.vega.\1 import"),
        # Fix voice module imports
        (r"from voice import", r"from src.vega.voice.voice_engine import"),
        (r"import voice\.([a-zA-Z_]+)", r"import src.vega.voice.\1"),
    ]

    # Apply all patterns
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Write back if changed
    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  ✅ Fixed imports in {filepath}")
        return True
    else:
        print(f"  ➡️  No changes needed in {filepath}")
        return False


def main():
    """Fix imports in all test files"""
    test_dir = "/home/ncacord/Vega2.0/tests"

    # Find all Python test files
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".py"):
                test_files.append(os.path.join(root, file))

    print(f"Found {len(test_files)} test files to process")

    fixed_count = 0
    for test_file in sorted(test_files):
        if fix_imports_in_file(test_file):
            fixed_count += 1

    print(
        f"\n✅ Fixed imports in {fixed_count} files out of {len(test_files)} total files"
    )


if __name__ == "__main__":
    main()
