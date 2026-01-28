# Copyright (c) WSmart-Route. All rights reserved.
"""
Centralized benchmark runner for WSmart-Route.
Executes all benchmark scripts in the logic/benchmark directory.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path):
    """Run a python script as a module and return its status."""
    module_name = script_path.replace("/", ".").replace(".py", "")
    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARK: {module_name}")
    print(f"{'='*60}\n")

    start_time = time.time()
    try:
        # We use subprocess to ensure a clean environment for each benchmark
        # and to handle any potential crashes gracefully.
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=os.getcwd(),
            capture_output=False, # Print directly to console
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {module_name} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {module_name} exited with code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Could not run {module_name}: {e}")
        return False

def main():
    # Set PYTHONPATH to root to ensure relative imports work
    os.environ["PYTHONPATH"] = os.getcwd()

    benchmark_dir = Path("logic/benchmark")

    # List of scripts to run in order
    # Note: excluding __init__.py and this script itself
    scripts = [
        "logic/benchmark/baseline_benchmarks.py",
        "logic/benchmark/benchmark_ls.py",
        "logic/benchmark/benchmark_policies.py",
        "logic/benchmark/benchmark_suite.py",
        "logic/benchmark/neural_benchmarks.py",
    ]

    print(f"Starting WSmart-Route Benchmark Suite")
    print(f"Directory: {benchmark_dir}")
    print(f"Found {len(scripts)} benchmark scripts to execute.\n")

    success_count = 0
    total_start = time.time()

    for script in scripts:
        if run_script(script):
            success_count += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Passed: {success_count}/{len(scripts)}")
    print(f"{'='*60}")

    if success_count < len(scripts):
        sys.exit(1)

if __name__ == "__main__":
    main()
