import os
import sys
import argparse
import traceback

from src.pipeline.test import run_wsr_simulator_test
from src.utils.arg_parser import (
    ConfigsParser, 
    add_test_sim_args, 
    validate_test_sim_args
)

# Ensure that the app root directory is on the path if needed
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="WSR Simulator Test Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    add_test_sim_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:])
        args = validate_test_sim_args(parsed_args)
        run_wsr_simulator_test(args)
    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        exit_code = 1
        print(str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)