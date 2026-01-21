"""
Test Runner for wsmart_bin_analysis Test Suite.

This script provides a convenient interface to run all tests or specific test modules
for the wsmart_bin_analysis testing suite.

Usage:
    # Run all tests
    python test_suite.py

    # Run specific test module
    python test_suite.py --module container

    # Run multiple modules
    python test_suite.py --module container extract simulation

    # Run with coverage report
    python test_suite.py --coverage

    # Run with verbose output
    python test_suite.py --verbose

    # Run specific test class
    python test_suite.py --class TestContainer

    # Run specific test method
    python test_suite.py --test test_container_init

    # List all available test modules
    python test_suite.py --list
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from .test_definitions import TEST_MODULES


class PyTestRunner:
    """
    Manages test execution using pytest, allowing for targeted testing of modules, classes, or methods.
    """

    def __init__(self, test_dir: str = "tests"):
        """
        Initialize the runner.

        Args:
            test_dir (str): Directory containing the test files. Defaults to 'tests'.
        """
        self.test_dir = Path(test_dir)
        self.available_modules = self._discover_test_modules()

    def _discover_test_modules(self) -> List[str]:
        """
        Discover all files starting with 'test_' in the test directory.

        Returns:
            List[str]: List of test module stems (filenames without extension).
        """
        if not self.test_dir.exists():
            return []

        test_files = list(self.test_dir.glob("test_*.py"))
        return [f.stem for f in test_files]

    def _build_pytest_command(
        self,
        modules: Optional[List[str]] = None,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        markers: Optional[str] = None,
        failed_first: bool = False,
        maxfail: Optional[int] = None,
        capture: str = "auto",
        tb_style: str = "auto",
        parallel: bool = False,
        keyword: Optional[str] = None,
    ) -> List[str]:
        """
        Construct the pytest command line arguments based on provided filters and options.

        Args:
            modules (Optional[List[str]]): List of short names or filenames of modules to test.
            test_class (Optional[str]): Name of a specific test class to run.
            test_method (Optional[str]): Name of a specific test method to run.
            verbose (bool): Whether to run pytest in verbose mode.
            coverage (bool): Whether to generate a coverage report.
            markers (Optional[str]): Pytest marker filter (e.g., 'not slow').
            failed_first (bool): Run previously failed tests first.
            maxfail (Optional[int]): Stop after N failures.
            capture (str): Stdout/stderr capture mode.
            tb_style (str): Traceback print style.
            parallel (bool): Run tests in parallel (autodetects core count).
            keyword (Optional[str]): Only run tests matching the given expression.

        Returns:
            List[str]: The complete list of command line arguments for subprocess.run.
        """
        cmd = ["pytest"]

        # Determine test targets
        if test_method:
            # Specific test method (requires module or searches all)
            if modules:
                for module in modules:
                    test_file = TEST_MODULES.get(module, f"test_{module}.py")
                    target = str(self.test_dir / test_file)
                    if test_class:
                        cmd.append(f"{target}::{test_class}::{test_method}")
                    else:
                        cmd.append(f"{target}::-k {test_method}")
            else:
                cmd.append(str(self.test_dir))
                if test_class:
                    cmd.extend(["-k", f"{test_class} and {test_method}"])
                else:
                    cmd.extend(["-k", test_method])

        elif test_class:
            # Specific test class
            if modules:
                for module in modules:
                    test_file = TEST_MODULES.get(module, f"test_{module}.py")
                    cmd.append(f"{self.test_dir / test_file}::{test_class}")
            else:
                cmd.append(str(self.test_dir))
                cmd.extend(["-k", test_class])

        elif modules:
            # Specific modules
            for module in modules:
                test_file = TEST_MODULES.get(module, f"test_{module}.py")
                cmd.append(str(self.test_dir / test_file))

        else:
            # All tests
            cmd.append(str(self.test_dir))

        # Add pytest options
        if verbose:
            cmd.append("-v")

        if coverage:
            cmd.extend(
                [
                    "--cov=configs_parser",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                ]
            )

        if markers:
            cmd.extend(["-m", markers])

        if failed_first:
            cmd.append("--ff")

        if maxfail:
            cmd.extend(["--maxfail", str(maxfail)])

        if capture != "auto":
            cmd.append(f"--capture={capture}")

        if tb_style != "auto":
            cmd.append(f"--tb={tb_style}")

        if parallel:
            # Requires pytest-xdist
            cmd.extend(["-n", "auto"])

        if keyword:
            cmd.extend(["-k", keyword])

        return cmd

    def run_tests(self, **kwargs) -> int:
        """
        Execute tests by calling pytest as a subprocess.

        Args:
            **kwargs: See _build_pytest_command for available options.

        Returns:
            int: The return code from the pytest process.
        """
        cmd = self._build_pytest_command(**kwargs)

        print(f"Running command: {' '.join(cmd)}")
        print("=" * 80)

        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except FileNotFoundError:
            print("Error: pytest not found. Please install it with: pip install pytest")
            return 1
        except KeyboardInterrupt:
            print("\nTest execution interrupted by user")
            return 130

    def list_modules(self) -> None:
        """List all discovered and predefined test modules to stdout."""
        print("\n" + "=" * 80)
        print("Available Test Modules:")
        print("=" * 80)

        if not self.available_modules:
            print("No test modules found in the test directory.")
            return

        # Predefined modules
        print("\nPredefined modules (can use short names):")
        for short_name, filename in sorted(TEST_MODULES.items()):
            status = "✓" if filename.replace(".py", "") in self.available_modules else "✗"
            print(f"  {status} {short_name:15} -> {filename}")

        # Discovered modules not in predefined list
        discovered_only = set(self.available_modules) - set(f.replace(".py", "") for f in TEST_MODULES.values())
        if discovered_only:
            print("\nAdditional discovered modules:")
            for module in sorted(discovered_only):
                print(f"  ✓ {module}.py")

        print("\n" + "=" * 80)

    def list_tests(self, module: Optional[str] = None) -> None:
        """
        List all individual test cases found in a specific module or across all modules.

        Args:
            module (Optional[str]): Module name to inspect. If None, inspects all.
        """
        cmd = ["pytest", "--collect-only", "-q"]

        if module:
            test_file = TEST_MODULES.get(module, f"test_{module}.py")
            cmd.append(str(self.test_dir / test_file))
        else:
            cmd.append(str(self.test_dir))

        print("Collecting tests...")
        print("=" * 80)
        subprocess.run(cmd)
