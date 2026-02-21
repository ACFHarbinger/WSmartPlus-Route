import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.tracking.logging.log_utils import output_stats


def aggregate_final_results(log_tmp: Any, cfg: Config, lock: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Aggregate results from all finished simulation samples.

    Args:
        log_tmp: Manager dict of policy -> list of metric lists.
        cfg: Root configuration.
        lock: Multiprocessing lock.
    """
    sim = cfg.sim
    policies = sim.full_policies

    if sim.n_samples > 1:
        if sim.resume:
            return output_stats(  # type: ignore[call-arg, misc]
                udef.ROOT_DIR,  # type: ignore[arg-type]
                sim.days,
                sim.graph.num_loc,
                sim.output_dir,
                sim.graph.area,
                sim.n_samples,
                policies,
                udef.SIM_METRICS,
                lock=lock,
            )
        else:
            log: Dict[str, Any] = {}
            log_std: Dict[str, Any] = {}
            log_full: Dict[str, List[List[float]]] = defaultdict(list)

            # Extract list from Manager objects
            for key, val in log_tmp.items():
                log_full[key].extend(val)

            for pol in policies:
                if log_full[pol]:
                    log[pol] = [statistics.mean(v) for v in zip(*log_full[pol])]
                    log_std[pol] = [statistics.stdev(v) if len(log_full[pol]) > 1 else 0.0 for v in zip(*log_full[pol])]
                else:
                    log[pol] = [0.0] * len(udef.SIM_METRICS)
                    log_std[pol] = [0.0] * len(udef.SIM_METRICS)
            return log, log_std
    else:
        log = {pol: res[0] for pol, res in log_tmp.items() if res}
        return log, None
