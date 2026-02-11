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

from logic.src.interfaces import ITraversable
from logic.src.utils.logging.log_utils import final_simulation_summary

from ..checkpoints import CheckpointError, checkpoint_manager
from ..day_context import SimulationDayContext, run_day
from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class RunningState(SimState):
    """State handles the day-by-day simulation loop."""

    def handle(self, ctx: SimulationContext) -> None:
        """Handle the day-by-day simulation loop."""
        opts = ctx.opts
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
                self._run_simulation_days(ctx, iterator, hook, realtime_log_path)

            logger.info(f"Simulation loop complete. Processed {opts['days']} days.")
            from .finishing import FinishingState

            ctx.transition_to(FinishingState())

        except CheckpointError as e:
            ctx.result = e.error_result
            if opts.get("print_output") and ctx.result:
                final_simulation_summary(ctx.result, ctx.policy, opts["n_samples"])
            ctx.transition_to(None)

    def _run_simulation_days(self, ctx, iterator, hook, realtime_log_path):
        for day in iterator:
            hook.before_day(day)

            current_policy_config = self._get_current_policy_config(ctx)
            day_context = self._create_day_context(ctx, day, current_policy_config, realtime_log_path)

            day_context = run_day(day_context)
            ctx.execution_time = time.process_time() - ctx.tic

            self._update_ctx_from_day_context(ctx, day_context)
            self._update_metrics(ctx, day, day_context.output_dict, day_context.daily_log)

            hook.after_day(ctx.execution_time)
            if ctx.overall_progress:
                ctx.overall_progress.update(1)

    def _get_current_policy_config(self, ctx):
        current_policy_config = {}  # type: ignore[var-annotated]
        for key, cfg in (ctx.config or {}).items():
            if key in ctx.policy:
                if isinstance(cfg, list):
                    for item in cfg:
                        if isinstance(item, ITraversable):
                            current_policy_config.update(item)
                elif isinstance(cfg, ITraversable):
                    current_policy_config.update(cfg)

        if "hgs" in ctx.pol_strip and "hgs_alns" not in ctx.pol_strip and ctx.vehicle_capacity is not None:
            if "hgs" not in current_policy_config:
                current_policy_config["hgs"] = {}
            if "capacity" not in current_policy_config["hgs"]:
                current_policy_config["hgs"]["capacity"] = ctx.vehicle_capacity
        return current_policy_config

    def _create_day_context(self, ctx, day, current_policy_config, realtime_log_path):
        assert ctx.dist_tup is not None
        (distance_matrix, paths_between_states, dm_tensor, distancesC) = ctx.dist_tup

        return SimulationDayContext(
            graph_size=ctx.opts["size"],
            full_policy=ctx.policy,
            policy=ctx.pol_strip,
            policy_name=ctx.pol_name or "",
            engine=ctx.pol_engine,
            threshold=ctx.pol_threshold,
            bins=ctx.bins,
            new_data=ctx.new_data,
            coords=ctx.coords,
            sample_id=ctx.sample_id,
            overflows=ctx.overflows,
            day=day,
            model_env=ctx.model_env,
            model_ls=ctx.model_tup or (None, None),
            n_vehicles=ctx.opts["n_vehicles"],
            area=ctx.opts["area"],
            realtime_log_path=realtime_log_path,
            waste_type=ctx.opts["waste_type"],
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
            config=current_policy_config,
            cost_weight=ctx.opts.get("cost_weight", 1.0),
            waste_weight=ctx.opts.get("waste_weight", 1.0),
            overflow_penalty=ctx.opts.get("overflow_penalty", 1.0),
        )

    def _update_ctx_from_day_context(self, ctx, day_context):
        ctx.new_data = day_context.new_data
        ctx.coords = day_context.coords
        ctx.bins = day_context.bins
        ctx.overflows = day_context.overflows
        ctx.cached = day_context.cached

        if ctx.counter:
            with ctx.counter.get_lock():
                ctx.counter.value += 1

    def _update_metrics(self, ctx, day, output_dict, dlog):
        if "am" in ctx.pol_strip or "transgcn" in ctx.pol_strip:
            if ctx.pol_strip not in ctx.attention_dict:
                ctx.attention_dict[ctx.pol_strip] = []
            ctx.attention_dict[ctx.pol_strip].append(output_dict)

        assert ctx.daily_log is not None
        if dlog is not None:
            for key, val in dlog.items():
                ctx.daily_log[key].append(val)

            if ctx.shared_metrics is not None:
                from logic.src.constants import METRICS

                cumulative_metrics = {
                    k: sum(v) for k, v in (ctx.daily_log or {}).items() if k in METRICS and k != "kg/km"
                }
                ctx.shared_metrics[f"{ctx.policy}_{ctx.sample_id}"] = {
                    "day": day,
                    "metrics": cumulative_metrics,
                    "daily_delta": dlog,
                    "policy": ctx.policy,
                    "sample_id": ctx.sample_id,
                }
