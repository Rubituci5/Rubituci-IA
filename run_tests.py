#!/usr/bin/env python3
"""
Test Runner for Entity Project

Runs all tests with proper configuration and reporting.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with pytest."""
    project_root = Path(__file__).parent

    # Test command with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-x",  # Stop on first failure
        # "--cov=.",  # Uncomment for coverage
        # "--cov-report=term-missing",
    ]

    print("=" * 60)
    print("RUNNING ENTITY PROJECT TESTS")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=project_root)

    print()
    print("=" * 60)
    if result.returncode == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return result.returncode


def run_specific_test(test_file):
    """Run a specific test file."""
    project_root = Path(__file__).parent

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
    ]

    print(f"Running {test_file}...")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        sys.exit(run_specific_test(sys.argv[1]))
    else:
        # Run all tests
        sys.exit(run_tests())