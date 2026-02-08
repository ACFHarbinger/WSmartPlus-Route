"""running.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import running
    """
from __future__ import annotations

import os
import time
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Optional, cast

from loguru import logger
from tqdm import tqdm

from logic.src.utils.logging.log_utils import final_simulation_summary

from ..checkpoints import CheckpointError, checkpoint_manager
from ..day_context import SimulationDayContext, run_day
from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class RunningState(SimState):
    """State handles the day-by-day simulation loop."""

    def handle(self, ctx: SimulationContext) -> None:
        """Handle.

        Args:
            ctx (SimulationContext): Description of ctx.
        """
        opts = ctx.opts
        desc = f"{ctx.policy} #{ctx.sample_id}"
        realtime_log_path = os.path.join(
            ctx.results_dir,
            f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl",
        )

        ctx.tic = time.process_time() + ctx.run_time

        try:
            assert ctx.checkpoint is not None
            with checkpoint_manager(ctx.checkpoint, opts["checkpoint_days"], ctx.get_current_state_tuple) as hook:
                hook.set_timer(ctx.tic)

                iterator = range(ctx.start_day, opts["days"] + 1)
                if not opts["no_progress_bar"]:
                    iterator = tqdm(
                        iterator,
                        desc=desc,
                        position=ctx.tqdm_pos + 1,
                        dynamic_ncols=True,
                        leave=False,
                        colour=ctx.colour,
                    )

                for day in iterator:
                    hook.before_day(day)
                    current_policy_config = {}
                    for key, cfg in (ctx.config or {}).items():
                        if key in ctx.policy:
                            if isinstance(cfg, list):
                                for item in cfg:
                                    if isinstance(item, dict):
                                        current_policy_config.update(item)
                            elif isinstance(cfg, dict):
                                current_policy_config.update(cfg)

                    if "hgs" in ctx.pol_strip and ctx.vehicle_capacity is not None:
                        if "hgs" not in current_policy_config:
                            current_policy_config["hgs"] = {}
                        if "capacity" not in current_policy_config["hgs"]:
                            current_policy_config["hgs"]["capacity"] = ctx.vehicle_capacity

                    assert ctx.dist_tup is not None
                    (distance_matrix, paths_between_states, dm_tensor, distancesC) = ctx.dist_tup

                    day_context = SimulationDayContext(
                        graph_size=opts["size"],
                        full_policy=ctx.policy,
                        policy=ctx.pol_strip,
                        policy_name=ctx.pol_name or "",
                        engine=ctx.pol_engine,
                        threshold=ctx.pol_threshold,
                        bins=ctx.bins,
                        new_data=ctx.new_data,
                        coords=ctx.coords,
                        run_tsp=opts["run_tsp"],
                        sample_id=ctx.sample_id,
                        overflows=ctx.overflows,
                        day=day,
                        model_env=ctx.model_env,
                        model_ls=ctx.model_tup or (None, None),
                        n_vehicles=opts["n_vehicles"],
                        area=opts["area"],
                        realtime_log_path=realtime_log_path,
                        waste_type=opts["waste_type"],
                        distpath_tup=ctx.dist_tup,
                        distance_matrix=distance_matrix,
                        distancesC=distancesC,
                        paths_between_states=paths_between_states,
                        dm_tensor=dm_tensor,
                        current_collection_day=ctx.current_collection_day,
                        cached=ctx.cached,
                        device=ctx.device,
                        lock=cast(Optional[Lock], ctx.lock),
                        hrl_manager=ctx.hrl_manager,
                        gate_prob_threshold=opts["gate_prob_threshold"],
                        mask_prob_threshold=opts["mask_prob_threshold"],
                        two_opt_max_iter=opts["two_opt_max_iter"],
                        config=current_policy_config,
                        cost_weight=opts.get("cost_weight", 1.0),
                        waste_weight=opts.get("waste_weight", 1.0),
                        overflow_penalty=opts.get("overflow_penalty", 1.0),
                    )

                    day_context = run_day(day_context)
                    ctx.execution_time = time.process_time() - ctx.tic

                    ctx.new_data = day_context.new_data
                    ctx.coords = day_context.coords
                    ctx.bins = day_context.bins
                    ctx.overflows = day_context.overflows
                    dlog = day_context.daily_log
                    output_dict = day_context.output_dict
                    ctx.cached = day_context.cached

                    if ctx.counter:
                        with ctx.counter.get_lock():
                            ctx.counter.value += 1

                    if "am" in ctx.pol_strip or "transgcn" in ctx.pol_strip:
                        if ctx.pol_strip not in ctx.attention_dict:
                            ctx.attention_dict[ctx.pol_strip] = []
                        ctx.attention_dict[ctx.pol_strip].append(output_dict)

                    assert ctx.daily_log is not None
                    if dlog is not None:
                        for key, val in dlog.items():
                            ctx.daily_log[key].append(val)

                    hook.after_day(ctx.execution_time)
                    if ctx.overall_progress:
                        ctx.overall_progress.update(1)

            logger.info(f"Simulation loop complete. Processed {opts['days']} days.")
            from .finishing import FinishingState

            ctx.transition_to(FinishingState())

        except CheckpointError as e:
            ctx.result = e.error_result
            if opts.get("print_output") and ctx.result:
                final_simulation_summary(ctx.result, ctx.policy, opts["n_samples"])
            ctx.transition_to(None)
