import statistics
from collections import defaultdict

import logic.src.constants as udef


def aggregate_final_results(log_tmp, opts, lock):
    """
    Aggregate results from all finished simulation samples.
    """
    from logic.src.utils.logging.log_utils import output_stats

    if opts["n_samples"] > 1:
        if opts["resume"]:
            return output_stats(  # type: ignore[call-arg, misc]
                udef.ROOT_DIR,  # type: ignore[arg-type]
                opts["days"],
                opts["size"],
                opts["output_dir"],
                opts["area"],
                opts["n_samples"],
                opts["policies"],
                udef.SIM_METRICS,
                lock=lock,
            )
        else:
            log = {}
            log_std = {}
            log_full = defaultdict(list)

            # Extract list from Manager objects
            for key, val in log_tmp.items():
                log_full[key].extend(val)

            for pol in opts["policies"]:
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
