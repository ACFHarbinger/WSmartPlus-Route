"""
GFACS Policy.

GFlowNet Ant Colony System policy with trajectory balance loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.deepaco import DeepACOPolicy
from logic.src.models.subnets.encoders.gfacs_encoder import GFACSEncoder
from logic.src.utils.functions.decoding import (
    batchify,
    get_log_likelihood,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering,
    unbatchify,
)


class GFACSPolicy(DeepACOPolicy):
    """
    GFACS policy based on NonAutoregressivePolicy.

    Introduced by Kim et al. (2024): https://arxiv.org/abs/2403.07041.
    Uses a Non-Autoregressive Graph Neural Network to generate heatmaps,
    which are then used to run Ant Colony Optimization (ACO) to construct solutions.

    Args:
        encoder: Encoder module. Can be passed by sub-classes.
        env_name: Name of the environment used to initialize embeddings.
        temperature: Temperature for the softmax during decoding. Defaults to 1.0.
        top_p: Top-p filtering threshold.
        top_k: Top-k filtering value.
        aco_class: Class representing the ACO algorithm to be used.
        aco_kwargs: Additional arguments to be passed to the ACO algorithm.
        train_with_local_search: Whether to train with local search. Defaults to True.
        n_ants: Number of ants to be used in the ACO algorithm.
        n_iterations: Number of iterations to run the ACO algorithm.
        multistart: Whether to use multistart decoding.
        k_sparse: Number of edges to keep for each node.
        **encoder_kwargs: Additional arguments to be passed to the encoder.
    """

    def __init__(
        self,
        encoder: Optional[GFACSEncoder] = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        aco_class: Optional[type] = None,
        aco_kwargs: Optional[Dict[str, Any]] = None,
        train_with_local_search: bool = True,
        n_ants: Optional[int | Dict[str, int]] = None,
        n_iterations: Optional[int | Dict[str, int]] = None,
        multistart: bool = False,
        k_sparse: Optional[int] = None,
        **encoder_kwargs,
    ) -> None:
        """
        Initialize GFACSPolicy.
        """
        aco_kwargs = aco_kwargs or {}

        if encoder is None:
            encoder_kwargs["z_out_dim"] = 2 if train_with_local_search else 1
            encoder_kwargs["k_sparse"] = k_sparse
            encoder = GFACSEncoder(env_name=env_name, **encoder_kwargs)

        # Convert optional dict/int to int for parent class
        n_ants_int = 20 if n_ants is None else (n_ants if isinstance(n_ants, int) else n_ants.get("train", 20))
        n_iterations_int = (
            1
            if n_iterations is None
            else (n_iterations if isinstance(n_iterations, int) else n_iterations.get("train", 1))
        )

        self.n_ants = (
            n_ants if isinstance(n_ants, dict) else {"train": n_ants_int, "val": n_ants_int, "test": n_ants_int}
        )
        self.n_iterations = (
            n_iterations
            if isinstance(n_iterations, dict)
            else {"train": n_iterations_int, "val": n_iterations_int, "test": n_iterations_int}
        )
        self.decode_type = "sampling"
        self.default_decoding_kwargs: Dict[str, Any] = {}
        self.train_with_local_search = train_with_local_search
        self.aco_class = aco_class
        self.aco_kwargs = aco_kwargs or {}

        super().__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            aco_class=aco_class,
            aco_kwargs=aco_kwargs,
            train_with_local_search=train_with_local_search,
            n_ants=n_ants_int,
            n_iterations=n_iterations_int,
            multistart=multistart,
            k_sparse=k_sparse,
        )

    def forward(  # type: ignore[override]
        self,
        td_initial: TensorDict,
        env: Optional[str | RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = True,
        return_hidden: bool = False,
        actions: Optional[torch.Tensor] = None,
        **decoding_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward method.

        During validation and testing, the policy runs the ACO algorithm to construct solutions.
        During training, uses trajectory balance loss.

        Args:
            td_initial: Initial TensorDict.
            env: Environment instance or name.
            phase: Current phase ('train', 'val', 'test').
            return_actions: Whether to return actions.
            return_hidden: Whether to return hidden states.
            actions: Optional pre-specified actions.

        Returns:
            Dictionary with rewards, actions, and other outputs.
        """
        n_ants = self.n_ants[phase]

        # Get heatmap and logZ from encoder (type: ignore because encoder is guaranteed to be GFACSEncoder)
        heatmap, _, logZ = self.encoder(td_initial)  # type: ignore[misc]

        decoding_kwargs.update(self.default_decoding_kwargs)
        decoding_kwargs.update({"num_starts": n_ants} if "multistart" in self.decode_type else {"num_samples": n_ants})

        # Instantiate environment if needed
        if (phase != "train" or self.train_with_local_search) and (env is None or isinstance(env, str)):
            from logic.src.envs import get_env

            env_name = self.env_name if env is None else (env if isinstance(env, str) else self.env_name)
            if env_name is None:
                raise ValueError("env_name must be set either in __init__ or forward")
            env = get_env(env_name)

        if phase == "train":
            # Encoder: get encoder output and initial embeddings from initial state
            if self.train_with_local_search:
                logZ, ls_logZ = logZ[:, [0]], logZ[:, [1]]
            else:
                logZ = logZ[:, [0]]
                ls_logZ = None  # Initialize to avoid unbound error

            logprobs, actions, td, env = self.common_decoding(
                decode_type=self.decode_type,
                td=td_initial.clone(),
                env=env,
                heatmap=heatmap,
                phase=phase,
                num_starts=n_ants,
                **decoding_kwargs,
            )

            outdict = {
                "logZ": logZ,
                "reward": unbatchify(env.get_reward(td, actions), n_ants),
                "log_likelihood": unbatchify(get_log_likelihood(logprobs, None, td.get("mask", None), True), n_ants),
            }

            if return_actions:
                outdict["actions"] = unbatchify(actions, n_ants)

            # Local search
            if self.train_with_local_search:
                aco = self.aco_class(heatmap, n_ants=n_ants, **self.aco_kwargs)
                ls_actions, ls_reward = aco.local_search(batchify(td_initial, n_ants), env, actions, decoding_kwargs)
                ls_decoding_kwargs = decoding_kwargs.copy()
                ls_decoding_kwargs["top_k"] = 0  # This should be 0, otherwise logprobs can be -inf
                ls_logprobs, ls_actions, td, env = self.common_decoding(
                    decode_type="evaluate",
                    td=td_initial.clone(),
                    env=env,
                    heatmap=heatmap,
                    actions=ls_actions,
                    **ls_decoding_kwargs,
                )
                outdict.update(
                    {
                        "ls_logZ": ls_logZ,
                        "ls_reward": unbatchify(ls_reward, n_ants),
                        "ls_log_likelihood": unbatchify(
                            get_log_likelihood(ls_logprobs, None, td.get("mask", None), True),
                            n_ants,
                        ),
                    }
                )
                if return_actions:
                    outdict["ls_actions"] = unbatchify(ls_actions, n_ants)

            if return_hidden:
                outdict["hidden"] = heatmap

            return outdict

        heatmap /= self.temperature

        if self.top_k > 0:
            self.top_k = min(self.top_k, heatmap.size(-1))  # Safety check
            heatmap = modify_logits_for_top_k_filtering(heatmap, self.top_k)

        if self.top_p > 0:
            assert self.top_p <= 1.0, "top-p should be in (0, 1]."
            heatmap = modify_logits_for_top_p_filtering(heatmap, self.top_p)

        aco = self.aco_class(heatmap, n_ants=n_ants, **self.aco_kwargs)
        actions, iter_rewards = aco.run(td_initial, env, self.n_iterations[phase], decoding_kwargs)

        out = {"reward": iter_rewards[self.n_iterations[phase] - 1]}
        out.update({f"reward_{i:03d}": iter_rewards[i] for i in range(self.n_iterations[phase])})
        if return_actions:
            out["actions"] = actions

        return out
