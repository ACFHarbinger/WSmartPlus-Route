"""
Parallel execution utilities.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Callable, List, Optional, Tuple

from tqdm import tqdm


def run_all_in_pool(
    func: Callable[..., Any],
    directory: str,
    dataset: List[Any],
    *,
    cpus: Optional[int] = None,
    offset: int = 0,
    n: Optional[int] = None,
    progress_bar_mininterval: float = 0.1,
    use_multiprocessing: bool = True,
) -> Tuple[List[Any], int]:
    """
    Runs a function over a dataset in parallel using multiprocessing or threading.

    Args:
        func: The function to execute.
        directory: Directory context for the function.
        dataset: List of problem instances.
        cpus: Number of CPUs to use. Defaults to all available.
        offset: Starting offset in the dataset. Defaults to 0.
        n: Maximum number of instances to process. Defaults to all.
        progress_bar_mininterval: Minimum interval for progress bar updates. Defaults to 0.1.
        use_multiprocessing: Whether to use process pool or thread pool. Defaults to True.

    Returns:
        tuple: (results, num_cpus)
    """
    num_cpus = (os.cpu_count() or 1) if cpus is None else cpus
    w = len(str(len(dataset) - 1))

    ds = dataset[offset : (offset + n if n is not None else len(dataset))]
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
