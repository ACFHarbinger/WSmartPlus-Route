"""ZenML simulation pipeline for WSmart-Route.

Wraps :func:`~logic.src.pipeline.features.test.engine.run_wsr_simulator_test`
in a ZenML pipeline with **per-policy fan-out** steps:

1. **prepare_sim_config** — serialise the Hydra config to a plain dict.
2. **run_policy_step** (×N) — one step per policy, each with an
   ``experiment_tracker`` decorator so ZenML manages an MLflow run.
   Steps are chained via a ``batch_id: str`` pass-through to create a
   sequential DAG dependency.
3. **aggregate_sim_results** — collect ``batch_id`` from the last policy
   step and log a summary.

The pipeline is invoked from
:func:`~logic.src.pipeline.features.test.engine._run_sim_via_zenml`
when ``cfg.tracking.zenml_enabled`` is ``True``.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any, Dict, List

from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Lazy ZenML imports
# ---------------------------------------------------------------------------

_ZENML_AVAILABLE = False
with contextlib.suppress(ImportError):
    from zenml import pipeline as zenml_pipeline  # type: ignore[import-not-found]
    from zenml import step  # type: ignore[import-not-found]

    _ZENML_AVAILABLE = True


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

if _ZENML_AVAILABLE:

    @step  # type: ignore[misc]
    def prepare_sim_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through: makes the serialised config a ZenML artifact."""
        return config_dict

    @step(experiment_tracker="mlflow_tracker")  # type: ignore[misc]
    def run_policy_step(
        config_dict: Dict[str, Any],
        policy_name: str,
        batch_id: str,
    ) -> str:
        """Run the full simulation for a single policy inside a ZenML step.

        The ``experiment_tracker`` decorator causes ZenML to manage an MLflow
        run.  A :class:`ZenMLBridge` is attached so WSTracker metrics are
        dual-written.

        Args:
            config_dict: Serialised root config.
            policy_name: Name of the policy to run (used for step naming).
            batch_id: Pass-through identifier for DAG dependency chaining.

        Returns:
            The same *batch_id* (creates a sequential DAG edge to the next
            policy step).
        """
        from omegaconf import OmegaConf

        from logic.src.configs import Config
        from logic.src.pipeline.features.test.engine import run_wsr_simulator_test
        from logic.src.tracking.integrations.zenml_bridge import ZenMLBridge

        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(config_dict))
        cfg = OmegaConf.to_object(cfg)
        assert isinstance(cfg, Config)

        # Filter to only the requested policy
        sim = cfg.sim
        original_policies = list(sim.full_policies)
        sim.full_policies = [
            p for p in original_policies if (str(p) if not isinstance(p, dict) else list(p.keys())[0]) == policy_name
        ]

        bridge = ZenMLBridge()
        run_wsr_simulator_test(cfg, sinks=[bridge])
        return batch_id

    @step  # type: ignore[misc]
    def aggregate_sim_results(batch_id: str) -> str:
        """Terminal step — logs completion and returns the batch ID."""
        logger.info(f"All simulation policies completed (batch {batch_id})")
        return batch_id

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @zenml_pipeline(name="wsmart_route_simulation")  # type: ignore[misc]
    def _simulation_pipeline(
        config_dict: Dict[str, Any],
        policy_names: List[str],
    ) -> str:
        """Per-policy fan-out simulation pipeline."""
        cfg_art = prepare_sim_config(config_dict)
        batch_id = str(uuid.uuid4())[:8]

        # Sequential chain: each policy step depends on the previous one
        for policy_name in policy_names:
            batch_id = run_policy_step.with_options(  # type: ignore[union-attr]
                name=f"sim_{policy_name}",
            )(config_dict=cfg_art, policy_name=policy_name, batch_id=batch_id)

        return aggregate_sim_results(batch_id)


# ---------------------------------------------------------------------------
# Public entry point (called from engine.py)
# ---------------------------------------------------------------------------


def simulation_pipeline(cfg: Any) -> None:
    """Serialise *cfg* and launch the ZenML simulation pipeline.

    Args:
        cfg: Root Hydra :class:`~logic.src.configs.Config` object.

    Raises:
        ImportError: If ZenML is not installed.
    """
    if not _ZENML_AVAILABLE:
        raise ImportError("zenml is not installed — cannot run ZenML simulation pipeline")

    from omegaconf import OmegaConf

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    # Extract policy names from the config
    sim = cfg.sim
    policy_names: List[str] = [str(p) if not isinstance(p, dict) else list(p.keys())[0] for p in sim.full_policies]

    _simulation_pipeline(config_dict=config_dict, policy_names=policy_names)
