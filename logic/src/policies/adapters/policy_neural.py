from typing import Any, List, Tuple

import torch

from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.neural_agent import NeuralAgent
from logic.src.utils.functions.function import move_to


@PolicyRegistry.register("neural")
class NeuralPolicy(IPolicy):
    """
    Neural Policy wrapper.
    Executes deep reinforcement learning models.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the neural policy.
        """
        model_env = kwargs["model_env"]
        model_ls = kwargs["model_ls"]
        bins = kwargs["bins"]
        device = kwargs["device"]
        fill = kwargs["fill"]
        dm_tensor = kwargs["dm_tensor"]
        run_tsp = kwargs["run_tsp"]
        hrl_manager = kwargs.get("hrl_manager")
        gate_prob_threshold = kwargs.get("gate_prob_threshold", 0.5)
        mask_prob_threshold = kwargs.get("mask_prob_threshold", 0.5)
        two_opt_max_iter = kwargs.get("two_opt_max_iter", 0)

        agent = NeuralAgent(model_env)
        model_data, graph, profit_vars = model_ls

        # Construct cost weights
        cost_weights = {
            "waste": kwargs.get("w_waste", 1.0),
            "length": kwargs.get("w_length", 1.0),
            "overflows": kwargs.get("w_overflows", 1.0),
        }

        # Data preparation
        model_data["waste"] = torch.as_tensor(bins.c, dtype=torch.float32).unsqueeze(0)
        if "fill_history" in model_data:
            model_data["current_fill"] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)
        daily_data = move_to(model_data, device)

        tour, cost, output_dict = agent.compute_simulator_day(
            daily_data,
            graph,
            dm_tensor,
            profit_vars,
            run_tsp,
            hrl_manager=hrl_manager,
            waste_history=bins.get_level_history(device=device),
            threshold=gate_prob_threshold,
            mask_threshold=mask_prob_threshold,
            two_opt_max_iter=two_opt_max_iter,
            cost_weights=cost_weights,
        )
        return tour, cost, output_dict
