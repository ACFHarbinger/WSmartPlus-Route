"""
Parallel execution utilities.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Callable, List, Tuple

from tqdm import tqdm


def run_all_in_pool(
    func: Callable[..., Any],
    directory: str,
    dataset: List[Any],
    opts: Any,
    use_multiprocessing: bool = True,
) -> Tuple[List[Any], int]:
    """
    Runs a function over a dataset in parallel using multiprocessing or threading.

    Args:
        func: The function to execute.
        directory: Directory context for the function.
        dataset: List of problem instances.
        opts: Options including 'cpus', 'offset', 'n', 'progress_bar_mininterval'.
        use_multiprocessing: Whether to use process pool or thread pool. Defaults to True.

    Returns:
        tuple: (results, num_cpus)
    """
    num_cpus = (os.cpu_count() or 1) if getattr(opts, "cpus", None) is None else opts.cpus
    w = len(str(len(dataset) - 1))
    offset = getattr(opts, "offset", None)
    if offset is None:
        offset = 0

    progress_bar_mininterval = getattr(opts, "progress_bar_mininterval", 0.1)

    ds = dataset[offset : (offset + opts.n if getattr(opts, "n", None) is not None else len(dataset))]
    pool_cls = mp.Pool if use_multiprocessing and num_cpus > 1 else ThreadPool
    with pool_cls(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    func,
                    [(directory, str(i + offset).zfill(w), *problem) for i, problem in enumerate(ds)],
                ),
                total=len(ds),
                mininterval=progress_bar_mininterval,
            )
        )

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus
