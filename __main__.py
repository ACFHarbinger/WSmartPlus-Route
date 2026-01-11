#!/usr/bin/env python

"""
Entry Point for Package Execution.

This script allows the project to be executed as a package (e.g., `python -m WSmart-Route`).
It initializes the `ConfigsParser`, handles argument parsing and consistency validation,
and dispatches execution to the appropriate pipeline (Train, Test, MRL, HPO).

It acts as a lightweight wrapper around the core logic pipelines, ensuring that critical
modules like `test` and `train` can be invoked via a unified CLI interface.
"""

import os
import sys
import argparse
import traceback

from logic.src.pipeline.test import run_wsr_simulator_test
from logic.src.pipeline.train import run_training
from logic.src.utils.arg_parser import (
    ConfigsParser, 
    add_test_sim_args, validate_test_sim_args,
    add_train_args, validate_train_args,
    add_mrl_train_args, add_hp_optim_args
)

# Ensure that the app root directory is on the path if needed
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    exit_code = 0
    
    # Initialize the main parser
    parser = ConfigsParser(
        description="WSR Simulator: Integrated Train & Test Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Create subparsers to handle different commands (train, test, etc.)
    subparsers = parser.add_subparsers(help='Execution command', dest='command', required=True)

    # --- 1. Test Simulator Command ---
    add_test_sim_args(subparsers.add_parser('test_sim', help='Run WSR Simulator Tests'))

    # --- 2. Training Commands ---
    add_train_args(subparsers.add_parser('train', help='Train base RL model'))
    add_mrl_train_args(subparsers.add_parser('mrl_train', help='Train Meta-RL model'))
    add_hp_optim_args(subparsers.add_parser('hp_optim', help='Run Hyperparameter Optimization'))
    try:
        command, parsed_args = parser.parse_process_args(sys.argv[1:])
        
        # Dispatch based on the selected command
        if command == 'test_sim':
            # Validation specific to testing
            args = validate_test_sim_args(parsed_args)
            run_wsr_simulator_test(args)
        elif command in ['train', 'mrl_train', 'hp_optim']:
            # Validation specific to training (handles all 3 training variants)
            args = validate_train_args(parsed_args)
            run_training(args, command)
        else:
            # Fallback if an unknown command bypasses the parser
            raise argparse.ArgumentError(None, f"Unknown command: {command}")
    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"\nError: {e}", file=sys.stderr)
    except Exception as e:
        exit_code = 1
        print(str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)