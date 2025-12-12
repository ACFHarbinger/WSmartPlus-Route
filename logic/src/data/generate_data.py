import os
import sys
import torch
import random
import argparse
import traceback
import numpy as np

from logic.src.utils.definitions import ROOT_DIR
from logic.src.utils.arg_parser import (
    ConfigsParser, 
    add_gen_data_args, 
    validate_gen_data_args
)
from logic.src.utils.data_utils import check_extension, save_dataset
from .generate_problem_data import *
from .generate_simulator_data import generate_wsr_data


def generate_datasets(opts):
    # Set the random seed and execute the program
    random.seed(opts['seed'])
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])

    gamma_dists = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    distributions_per_problem = {
        'tsp': [None],
        'vrp': [None],
        'pctsp': [None],
        'op': ['empty', 'const', 'unif', 'dist', 'emp', *gamma_dists],
        'vrpp': ['empty', 'const', 'unif', 'dist', 'emp', *gamma_dists],
        'wcrp': ['empty', 'const', 'unif', 'dist', 'emp', *gamma_dists],
        'wcvrp': ['empty', 'const', 'unif', 'dist', 'emp', *gamma_dists],
        'pdp': [None]
    }

    # Define the problem distribution(s)
    if opts['problem'] == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts['problem']:
                distributions_per_problem[opts['problem']]
                if len(opts['data_distributions']) == 1 and opts['data_distributions'][0] == 'all'
                else [data_dist for data_dist in opts['data_distributions']]
        }

    # Generate the dataset(s)
    n_days = opts['n_epochs'] - opts['epoch_start'] if opts['dataset_type'] != 'train' else 0
    for problem, distributions in problems.items():
        datadir = os.path.join(ROOT_DIR, 'data', opts['data_dir'], problem) if opts['dataset_type'] in ['train_time', 'train'] \
                    else os.path.join(ROOT_DIR, 'data', "wsr_simulator", opts['data_dir'])
        try:
            os.makedirs(datadir, exist_ok=True) # Create directory for saving data
        except Exception:
            raise Exception("directories to save generated data files do not exist and could not be created")

        try:
            for dist in distributions or [None]:
                for size, graph in zip(opts['graph_sizes'], opts['focus_graphs']):
                    print("Generating '{}{}' ({}) dataset for the {} with {} locations{}{}".format(
                        opts['name'], n_days if n_days > 0 else "", 
                        opts['dataset_type'], problem.upper(), size, 
                        " and using '{}' as the instance distribution".format(dist),
                        ":" if n_days == 0 else "..."
                    ))
                    
                    if opts['dataset_type'] == 'test_simulator':
                        if 'filename' not in opts or opts['filename'] is None:
                            filename = os.path.join(datadir, 
                                "{}{}{}_{}{}_N{}_seed{}.pkl".format(opts['area'], size,
                                "_{}".format(dist) if dist is not None else "", 
                                opts['name'], n_days if n_days > 1 else "", 
                                opts['dataset_size'], opts['seed']))
                        else:
                            filename = check_extension(opts['filename'])
                        
                        dataset = generate_wsr_data(size, n_days, opts['dataset_size'], opts['area'], 
                                                    opts['waste_type'], dist, graph, opts['vertex_method'])
                        save_dataset(dataset, filename)
                    elif opts['dataset_type'] == 'train_time':
                        if 'filename' not in opts or opts['filename'] is None:
                            if opts['is_gaussian']:
                                filename = os.path.join(datadir, 
                                "{}{}{}_{}{}_seed{}_{}_{}.pkl".format(problem, size, 
                                "_{}".format(dist) if dist is not None else "",
                                opts['name'], n_days if n_days > 1 else "", 
                                opts['seed'], 'gaussian', opts['sigma']))
                            else:
                                filename = os.path.join(datadir, 
                                "{}{}{}_{}{}_seed{}.pkl".format(problem, size, 
                                "_{}".format(dist) if dist is not None else "",
                                opts['name'], n_days if n_days > 1 else "", opts['seed']))
                        else:
                            filename = check_extension(opts['filename'])

                        assert opts['f'] or not os.path.isfile(check_extension(filename)), \
                        "File already exists! Try running with -f option to overwrite."
                        if problem == "vrpp":
                            dataset = generate_vrpp_data(opts['dataset_size'], size, opts['waste_type'], dist, opts['area'], 
                                                        graph, opts['focus_size'], opts['vertex_method'], num_days=opts['n_epochs'])
                        elif problem == "wcrp":
                            dataset = generate_wcrp_data(opts['dataset_size'], size, opts['waste_type'], dist, opts['area'], 
                                                        graph, opts['focus_size'], opts['vertex_method'], num_days=opts['n_epochs'])
                        else:
                            assert problem == 'wcvrp'
                            dataset = generate_wcvrp_data(opts['dataset_size'], size, opts['waste_type'], dist, opts['area'], 
                                                        graph, opts['focus_size'], opts['vertex_method'], num_days=opts['n_epochs'])
                        save_dataset(dataset, filename)
                    else:
                        assert opts['dataset_type'] == 'train'
                        for epoch in range(opts['epoch_start'], opts['n_epochs']):
                            print("- Generating epoch {} data".format(epoch))
                            if 'filename' not in opts or opts['filename'] is None:
                                if opts['is_gaussian']:
                                    filename = os.path.join(datadir, 
                                    "{}{}{}_{}{}_seed{}_{}_{}.pkl".format(problem, size, 
                                    "_{}".format(dist) if dist is not None else "",
                                    opts['name'], epoch if opts['n_epochs'] > 1 else "", 
                                    opts['seed'], 'gaussian', opts['sigma']))
                                else:
                                    filename = os.path.join(datadir, 
                                    "{}{}{}_{}{}_seed{}.pkl".format(problem, size, 
                                    "_{}".format(dist) if dist is not None else "",
                                    opts['name'], epoch if opts['n_epochs'] > 1 else "", opts['seed']))
                            else:
                                filename = check_extension(opts['filename'])

                            assert opts['f'] or not os.path.isfile(check_extension(filename)), \
                            "File already exists! Try running with -f option to overwrite."
                            if problem == 'tsp':
                                dataset = generate_tsp_data(opts['dataset_size'], size,
                                opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == 'vrp':
                                dataset = generate_vrp_data(opts['dataset_size'], size,
                                opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == 'pctsp':
                                dataset = generate_pctsp_data(opts['dataset_size'], size, opts['penalty_factor'],
                                opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == "op":
                                dataset = generate_op_data(opts['dataset_size'], size, dist,
                                opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == "vrpp":
                                dataset = generate_vrpp_data(opts['dataset_size'], size, opts['waste_type'], 
                                dist, opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == "wcrp":
                                dataset = generate_wcrp_data(opts['dataset_size'], size, opts['waste_type'], 
                                dist, opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == "wcvrp":
                                dataset = generate_wcvrp_data(opts['dataset_size'], size, opts['waste_type'],
                                dist, opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            elif problem == 'pdp':
                                dataset = generate_pdp_data(opts['dataset_size'], size, opts['is_gaussian'],
                                opts['sigma'], opts['area'], graph, opts['focus_size'], opts['vertex_method'])
                            else:
                                assert False, "Unknown problem: {}".format(problem)
                            save_dataset(dataset, filename)
        except Exception as e:
            has_dists = len(distributions) >= 1 and distributions[0] is not None
            raise Exception("failed to generate data for problem {}{} due to {}".format(
                problem, f" {distributions}" if has_dists else "", repr(e)
            ))


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="Data Generator Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    add_gen_data_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:], "gen_data")
        args = validate_gen_data_args(parsed_args)
        generate_datasets(args)
    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)