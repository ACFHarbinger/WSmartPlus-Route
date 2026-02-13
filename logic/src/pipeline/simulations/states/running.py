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
        """Standardizes policy config resolution from structured context."""
        current_policy_config = {}  # type: ignore[var-annotated]

        # 1. START WITH GLOBAL CONFIGS (must_go, post_processing)
        if ctx.config:
            for g_key in ["must_go", "post_processing"]:
                if g_key in ctx.config:
                    current_policy_config[g_key] = ctx.config[g_key]

        # 2. ADD POLICY-SPECIFIC CONFIG FROM CONTEXT
        # This is the new primary path for structured configurations
        if ctx.policy_cfg:
            current_policy_config.update(ctx.policy_cfg)

        # 3. LEGACY FALLBACK: Match policy name in global config keys
        else:
            for key, cfg in (ctx.config or {}).items():
                if key in ctx.pol_name and isinstance(cfg, ITraversable):
                    current_policy_config.update(cfg)

        return current_policy_config

    def _create_day_context(self, ctx, day, current_policy_config, realtime_log_path):
        assert ctx.dist_tup is not None
        (distance_matrix, paths_between_states, dm_tensor, distancesC) = ctx.dist_tup

        return SimulationDayContext(
            graph_size=ctx.opts["size"],
            full_policy=ctx.policy,
            policy=ctx.policy,
            policy_name=ctx.pol_name,
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
