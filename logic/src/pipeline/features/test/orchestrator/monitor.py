import statistics
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict

import logic.src.constants as udef
from logic.src.constants import METRICS, SIM_METRICS
from logic.src.pipeline.callbacks.simulation_display import SimulationDisplay


def initialize_simulation_display(opts):
    """
    Initialize the simulation dashboard display if enabled.
    """
    if opts.get("no_progress_bar"):
        return None

    display = SimulationDisplay(policies=opts["policies"], n_samples=opts["n_samples"], total_days=opts["days"])
    display.start()
    return display


def process_display_updates(
    display: SimulationDisplay,
    shared_metrics: dict,
    log_tmp: dict,
    last_reported_days: dict,
    opts: dict,
    loop_tic: float,
    counter: Any,
):
    """
    Process real-time simulation metrics and update the dashboard display.
    """
    policy_updates = {}
    new_daily_data = []  # For chart updates

    policy_days_done: Dict[str, int] = defaultdict(int)
    policy_sample_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {k: [] for k in SIM_METRICS})  # type: ignore[assignment]
    policy_sample_counts: Dict[str, int] = defaultdict(int)

    # A. Process ACTIVE samples from shared_metrics
    for _key, data in shared_metrics.items():
        pol = data["policy"]
        sid = data["sample_id"]
        day = data["day"]
        metrics = data["metrics"]

        policy_days_done[pol] += day
        policy_sample_counts[pol] += 1

        for k in METRICS:
            if k in metrics:
                policy_sample_metrics[pol][k].append(metrics[k])
        policy_sample_metrics[pol]["days"].append(day)

        if (pol, sid) not in last_reported_days or last_reported_days[(pol, sid)] < day:
            delta = data.get("daily_delta", metrics)
            new_daily_data.append({"policy": pol, "day": day, "metrics": delta})
            last_reported_days[(pol, sid)] = day

    # B. Add COMPLETED samples from log_tmp
    for pol, results in log_tmp.items():
        for res in results:
            policy_sample_counts[pol] += 1
            policy_days_done[pol] += res[8]  # days index
            for i, k in enumerate(SIM_METRICS):
                policy_sample_metrics[pol][k].append(res[i])

    # 3. Calculate averages and update display
    elapsed_total = time.time() - loop_tic
    for pol in opts["policies"]:
        n_finished = len(log_tmp[pol])
        divisor = max(1, policy_sample_counts[pol])

        avg_metrics = {}
        for k in SIM_METRICS:
            vals = policy_sample_metrics[pol][k]
            if not vals:
                avg_metrics[k] = (0.0, 0.0)
                continue

            if k == "time" and sum(vals) == 0:
                m = elapsed_total / divisor
                s = 0.0
            else:
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0

            avg_metrics[k] = (m, s)

        policy_updates[pol] = {
            "metrics": avg_metrics,
            "completed": n_finished,
            "total_days_done": policy_days_done[pol],
        }

    display.update(counter.value, policy_updates, new_daily_data)


def monitor_tasks_until_complete(tasks, display, opts, counter, log_tmp):
    """
    Monitor task progress and update display until all tasks complete.
    """
    last_reported_days = {}  # type: ignore[var-annotated]
    loop_tic = time.time()

    while not all(task.ready() for task in tasks):
        if display:
            process_display_updates(
                display=display,
                shared_metrics=opts["shared_metrics"],
                log_tmp=log_tmp,
                last_reported_days=last_reported_days,
                opts=opts,
                loop_tic=loop_tic,
                counter=counter,
            )
        time.sleep(udef.PBAR_WAIT_TIME)


def collect_all_task_results(tasks):
    """
    Collect results from all tasks, logging any exceptions.
    """
    for task in tasks:
        try:
            task.get()
        except Exception as e:
            print(f"Task failed with exception: {e}")
            traceback.print_exc(file=sys.stdout)
