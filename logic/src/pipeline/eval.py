import os
import sys
import math
import time
import torch
import random
import argparse
import datetime
import traceback
import itertools
import numpy as np

from tqdm import tqdm
from logic.src.utils.arg_parser import (
    ConfigsParser, 
    add_eval_args, 
    validate_eval_args
)
from logic.src.utils.data_utils import save_dataset
from logic.src.utils.functions import move_to, load_model


mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)
    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]
    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args
    model, _ = load_model(opts['model'])
    val_size = opts['val_size'] // num_processes
    dataset = model.problem.make_dataset(
        filename=dataset_path, num_samples=val_size, offset=opts['offset'] + val_size * i,
        distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'],
        size=opts['graph_size'], focus_graph=opts['focus_graph'], focus_size=opts['focus_size'],
        area=opts['area'], dist_matrix_path=opts['dm_filepath'], number_edges=opts['edge_threshold'],
        waste_type=opts['waste_type'], dist_strat=opts['distance_method'], edge_strat=opts['edge_method']
    )
    device = torch.device("cuda:{}".format(i))
    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts, method):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts['model'], method)
    use_cuda = torch.cuda.is_available()
    if opts['multiprocessing']:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts['val_size'] % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))
    else:
        device = torch.device("cpu" if not use_cuda else "cuda:0")
        dataset = model.problem.make_dataset(
            filename=dataset_path, num_samples=opts['val_size'], offset=opts['offset'],
            distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'],
            size=opts['graph_size'], focus_graph=opts['focus_graph'], focus_size=opts['focus_size'],
            area=opts['area'], number_edges=opts['edge_threshold'], dist_matrix_path=opts['dm_filepath'],
            wtype=opts['waste_type'], edge_strat=opts['edge_method'], dist_strat=opts['distance_method']
        )
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts['eval_batch_size']
    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(datetime.timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts['model'])[0]).split(os.sep)[-2:])
    if opts['o'] is None:
        results_dir = os.path.join(opts['results_dir'], model.problem.NAME, dataset_basename)
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception:
            raise Exception("directories to save evaluation results do not exist and could not be created")

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts['decode_strategy'],
            width if opts['decode_strategy'] != 'greedy' else '',
            softmax_temp, opts['offset'], opts['offset'] + len(costs), ext
        ))
    else:
        out_file = opts['o']

    assert opts['f'] or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)
    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):
    model.to(device)
    model.eval()
    model.set_decode_type(
        "greedy" if opts['decode_strategy'] in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts['eval_batch_size'], pin_memory=True)
    results = []
    for batch in tqdm(dataloader, disable=opts['no_progress_bar']):
        batch = move_to(batch, device)
        start = time.time()
        with torch.no_grad():
            if opts['decode_strategy'] in ('sample', 'greedy'):
                if opts['decode_strategy'] == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts['eval_batch_size'] <= opts['max_calc_batch_size'], \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts['eval_batch_size'] > opts['max_calc_batch_size']:
                    assert opts['eval_batch_size'] == 1
                    assert width % opts['max_calc_batch_size'] == 0
                    batch_rep = opts['max_calc_batch_size']
                    iter_rep = width // opts['max_calc_batch_size']
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0

                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts['decode_strategy'] == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts['compress_mask'],
                    max_calc_batch_size=opts['max_calc_batch_size']
                )
        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp", "pdp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp", "vrpp", "wcvrp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                seq = None
                assert False, "Unknown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))
    return results


def run_evaluate_model(opts):
    # Set the random seed and execute the program
    random.seed(opts['seed'])
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    widths = opts['width'] if opts['width'] is not None else [0]
    for width in widths:
        for dataset_path in opts['datasets']:
            eval_dataset(dataset_path, width, opts['softmax_temperature'], opts)
    return


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="Evaluation Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    add_eval_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:], "eval")
        args = validate_eval_args(parsed_args)
        run_evaluate_model(args)
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