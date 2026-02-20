import time
from typing import Any, List, Tuple

import logic.src.constants as udef


def prepare_parallel_task_args(
    policies: List[str],
    n_samples: int,
    indices: List[Any],
    sample_idx_ls: List[List[int]],
) -> List[Tuple[Any, ...]]:
    """
    Prepare argument tuples for parallel task execution.

    Args:
        policies: Expanded policy name list.
        n_samples: Number of samples per policy.
        indices: Bin subset indices per sample.
        sample_idx_ls: Sample index lists per policy.
    """
    if n_samples > 1:
        return [(indices[sid], sid, pol_id) for pol_id in range(len(policies)) for sid in sample_idx_ls[pol_id]]
    else:
        return [(indices[0], 0, pol_id) for pol_id in range(len(policies))]


def print_execution_info(task_count: int, n_cores: int) -> None:
    """
    Print information about parallel execution configuration.
    """
    if n_cores > 1:
        print(f"Launching {task_count} WSmart Route simulations on {n_cores} CPU cores...")
        max_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT))
        proc_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.CORE_LOCK_WAIT_TIME))
        print(f"[INFO] Maximum lock wait time: {max_lock_timeout} ({proc_lock_timeout} per used thread)")
    else:
        print(f"Launching {task_count} WSmart Route simulations on a single CPU core...")
