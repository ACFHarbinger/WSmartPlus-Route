#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
    message="jax.tree_util.register_keypaths is deprecated")

import io
import os
import sys
import signal
import pprint
import traceback
import multiprocessing as mp
import logic.src.utils.definitions as udef

from logic.test import PyTestRunner
from logic.src.file_system import (
    perform_cryptographic_operations,
    update_file_system_entries, 
    delete_file_system_entries
)
from logic.src.pipeline.test import run_wsr_simulator_test
from logic.src.pipeline.eval import run_evaluate_model
from logic.src.pipeline.train import (
    train_reinforcement_learning, 
    train_reinforce_over_time, train_reinforce_epoch,
    train_meta_reinforcement_learning, hyperparameter_optimization
)
from logic.src.data.generate_data import generate_datasets
from logic.src.utils.arg_parser import parse_params 
from gui.src.app import run_app_gui, launch_results_window


def run_test_suite(opts):
    try:
        # Initialize test runner
        runner = PyTestRunner(test_dir=opts['test_dir'])
        
        # Handle information commands
        if opts['list']:
            runner.list_modules()
            return 0
        
        if opts['list_tests']:
            runner.list_tests(opts['module'][0] if opts['module'] else None)
            return 0
        
        # Run tests
        return runner.run_tests(
            modules=opts['module'],
            test_class=opts['test_class'],
            test_method=opts['test_method'],
            verbose=opts['verbose'],
            coverage=opts['coverage'],
            markers=opts['markers'],
            failed_first=opts['failed_first'],
            maxfail=opts['maxfail'],
            capture=opts['capture'],
            tb_style=opts['tb'],
            parallel=opts['parallel'],
            keyword=opts['keyword']
        )
    except Exception as e:
        raise Exception(f"failed to run test suite due to {repr(e)}")


def pretty_print_args(comm, opts, inner_comm=None):
    try:
        # Capture the pprint output
        printer = pprint.PrettyPrinter(width=1, indent=1, sort_dicts=False)
        buffer = io.StringIO()
        printer._stream = buffer # Redirect PrettyPrinter's internal stream
        printer.pprint(opts)
        output = buffer.getvalue()

        # Pretty print the run options
        lines = output.splitlines()
        lines[0] = lines[0].lstrip('{')
        lines[-1] = lines[-1].rstrip('}')
        formatted = comm + "{}".format(f' {inner_comm}' if inner_comm is not None else "") + \
            ": {\n" + "\n".join(f" {line}" for line in lines) + "\n}"
        print(formatted, end="\n\n")
    except Exception as e:
        raise Exception(f"failed to pretty print arguments due to {repr(e)}")


def main(args):
    comm, opts = args
    exit_code = 0
    try:
        if isinstance(comm, tuple) and len(comm) > 1:
            comm, inner_comm = comm
            pretty_print_args(comm, opts, inner_comm)
            assert comm == 'file_system'
            if inner_comm == 'update':
                update_file_system_entries(opts)
            elif inner_comm == 'delete':
                delete_file_system_entries(opts)
            else:
                assert inner_comm == 'cryptography'
                perform_cryptographic_operations(opts)
        else:
            inner_comm = None
            pretty_print_args(comm, opts, inner_comm)
            if comm == 'gui':
                exit_code = run_app_gui(opts)
            elif comm == 'test_suite':
                run_test_suite(opts)       
            else:
                if comm == 'train':
                    #if opts['rl_algorithm'] == 'reinforce':
                    train_func = train_reinforce_over_time if opts['train_time'] else train_reinforce_epoch
                    #else:
                    #    raise ValueError(f"Unknown reinforcement learning algorithm: {opts['rl_algorithm']}")
                    train_reinforcement_learning(opts, train_func)
                elif comm == 'mrl_train':
                    train_meta_reinforcement_learning(opts)
                elif comm == 'hp_optim':
                    hyperparameter_optimization(opts)
                elif comm == 'gen_data':
                    generate_datasets(opts)
                elif comm == 'eval':
                    run_evaluate_model(opts)
                elif comm == 'test_sim':
                    if opts['real_time_log']:
                        mp.set_start_method("spawn", force=True)
                        simulation_process = mp.Process(
                            target=run_wsr_simulator_test,
                            args=(opts,)
                        )
                        log_path = os.path.join(udef.ROOT_DIR, "assets", opts['output_dir'], 
                            str(opts['days']) + "_days", str(opts['area']) + '_' + str(opts['size']), 
                            f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl")
                        simulation_process.start()

                        # Define the handler function that terminates the subprocess
                        def handle_interrupt(signum, frame):
                            print("\nCtrl+C received. Terminating simulation process...")
                            if simulation_process.is_alive():
                                simulation_process.terminate()
                                simulation_process.join()
                            # Force the GUI application to quit gracefully
                            sys.exit(0)

                        # Register the handler only for this scope
                        original_sigint_handler = signal.getsignal(signal.SIGINT)
                        signal.signal(signal.SIGINT, handle_interrupt)
                        
                        try:
                            # 3. Blocking GUI call
                            exit_code = launch_results_window(opts['policies'], log_path)
                            
                        except SystemExit as e:
                            # Catch the sys.exit(0) from the handler, if triggered.
                            exit_code = e.code
                            
                        finally:
                            # 4. Restore original handler and clean up the process
                            signal.signal(signal.SIGINT, original_sigint_handler)
                            
                            if simulation_process.is_alive():
                                print("GUI closed. Terminating lingering simulation process.")
                                simulation_process.terminate()
                                simulation_process.join()
                    else:
                        run_wsr_simulator_test(opts)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print('\n' + str(e))
        exit_code = 1
    finally:
        print("\nFinished {}{} command execution with exit code: {}".format(
            comm, f" ({inner_comm}) " if inner_comm is not None else "", exit_code
        ))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


if __name__ =="__main__":
    main(parse_params())