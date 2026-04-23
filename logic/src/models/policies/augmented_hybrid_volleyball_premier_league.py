"""Vectorized Augmented Hybrid Volleyball Premier League (AHVPL) Policy.

This module implements the AHVPL algorithm, which fuses the population-based
Volleyball Premier League (VPL) framework with Hybrid Genetic Search (HGS)
operators and diversity management. It utilizes parallel ACO for initial
construction, HGS ordered crossover for learning, and ALNS for coaching/local
search.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.policies.hgs_core.crossover import (
    vectorized_ordered_crossover,
)
from logic.src.models.policies.hgs_core.population import VectorizedPopulation
from logic.src.models.policies.hybrid_volleyball_premier_league import (
    VectorizedHVPL,
)
from logic.src.models.policies.shared.linear import vectorized_linear_split


class VectorizedAHVPL(VectorizedHVPL):
    """Augmented Hybrid VPL Policy.

    Combines the competition/coaching dynamics of VPL with the robust genetic
    recombination and diversity-aware survivor selection of HGS. All phases
    are vectorized for high-throughput GPU execution.

    Attributes:
        crossover_rate: Probability of applying genetic crossover.
        elite_size: Number of top individuals protected during selection.
    """

    def __init__(
        self,
        env_name: str,
        n_teams: int = 10,
        max_iterations: int = 20,
        sub_rate: float = 0.2,
        time_limit: float = 60.0,
        aco_iterations: int = 1,
        alns_iterations: int = 50,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AHVPL solver.

        Args:
            env_name: Identifier for the problem environment.
            n_teams: Population size (teams) per batch instance.
            max_iterations: Number of league seasons (cycles).
            sub_rate: Percentage of non-improving teams to replace via ACO.
            time_limit: Wall-clock timeout for the entire search in seconds.
            aco_iterations: Construction phase iterations.
            alns_iterations: Coaching phase (local search) iterations.
            crossover_rate: Frequency of genetic recombination.
            elite_size: Protected group size for survivor selection.
            device: Target hardware.
            generator: Torch RNG instance.
            rng: Standard library RNG instance.
            **kwargs: Additional hyperparameters.
        """
        super().__init__(
            env_name=env_name,
            n_teams=n_teams,
            max_iterations=max_iterations,
            sub_rate=sub_rate,
            time_limit=time_limit,
            aco_iterations=aco_iterations,
            alns_iterations=alns_iterations,
            device=device,
            generator=generator,
            rng=rng,
            **kwargs,
        )
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for serialization, stripping non-picklable RNG.

        Returns:
            Dict[str, Any]: Attribute map safe for persistent storage.
        """
        state = self.__dict__.copy()
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore search state and re-initialize hardware-local RNG.

        Args:
            state: Serialized dictionary of solver attributes.
        """
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the augmented hybrid meta-heuristic search.

        Orchestrates ACO initialization, HGS-based learning (crossover),
        ALNS-based coaching, and HGS-style population management across all batches.

        Args:
            td: State bundle including locations and node demands.
            env: Relevant environment instance.
            strategy: Selection strategy (ignored).
            num_starts: Number of restarts (not applicable).
            max_steps: Sequence length limit (ignored).
            phase: Current execution phase ('train', 'val', or 'test').
            return_actions: Whether to include results in the output.
            **kwargs: Runtime overrides.

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - actions (torch.Tensor): Padded action sequences.
                - reward (torch.Tensor): Calculated reward for the solution.
                - cost (torch.Tensor): Raw objective value from the engine.
                - log_likelihood (torch.Tensor): Zero vector (AHVPL is non-pi).
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device
        start_time = time.time()

        # 1. Setup Data
        dist_matrix, waste, capacity = self._setup_data(td)
        num_nodes = dist_matrix.shape[1]

        # 2. League Initialization (ACO Construction)
        tau = torch.ones((batch_size, num_nodes, num_nodes), device=device) * 0.1
        eta = 1.0 / (dist_matrix + 1e-6)

        population_giant = self._aco_construct(dist_matrix, tau, eta, self.n_teams)

        expanded_dist = dist_matrix.repeat_interleave(self.n_teams, dim=0)
        expanded_waste = waste.repeat_interleave(self.n_teams, dim=0)
        expanded_capacity = capacity.repeat_interleave(self.n_teams, dim=0)

        pop_manager = VectorizedPopulation(self.n_teams, device, self.generator)

        flat_giant = population_giant.view(batch_size * self.n_teams, -1)
        _, init_costs = vectorized_linear_split(flat_giant, expanded_dist, expanded_waste, expanded_capacity)
        pop_manager.initialize(population_giant, init_costs.view(batch_size, self.n_teams), self.elite_size)

        best_tours = population_giant[:, 0].clone()
        best_costs = torch.full((batch_size,), float("inf"), device=device)

        # 3. League Season
        for _iter in range(self.max_iterations):
            if time.time() - start_time > self.time_limit:
                break

            # 4. Learning Phase: Genetic Crossover
            p1, p2 = pop_manager.get_parents(n_offspring=self.n_teams)

            if torch.rand(1, device=device, generator=self.generator).item() < self.crossover_rate:
                offspring_giant = vectorized_ordered_crossover(
                    p1.view(batch_size * self.n_teams, -1),
                    p2.view(batch_size * self.n_teams, -1),
                    device=device,
                    generator=self.generator,
                ).view(batch_size, self.n_teams, -1)
            else:
                offspring_giant = p1

            # 5. Coaching Phase: ALNS Improvement
            instance_costs, coached_routes_list = self._coaching_phase(
                offspring_giant, expanded_dist, expanded_waste, expanded_capacity
            )

            # 6. Global Competition: Update Best
            improved = self._global_competition(
                instance_costs,
                mentored_routes_list=coached_routes_list,
                best_tours=best_tours,
                best_costs=best_costs,
                num_nodes=num_nodes,
            )

            # 7. Survivor Selection: Diversity-Aware Population Update
            coached_giant = self._routes_to_giant_tours(
                coached_routes_list,
                batch_size,
                self.n_teams,
                num_nodes - 1,
                offspring_giant,
            )
            pop_manager.add_individuals(coached_giant, instance_costs, self.elite_size)

            # 8. Pheromone Update
            if improved.any():
                tau = self._update_pheromones(tau, best_tours, best_costs)

            # 9. VPL Substitution Phase
            if self.sub_rate > 0:
                self._ahvpl_substitution(pop_manager, dist_matrix, waste, capacity, tau, eta)

        # 10. Final packaging
        return self._format_output(td, best_tours, dist_matrix, waste, capacity, env=env)

    def _ahvpl_substitution(
        self,
        pop_manager: VectorizedPopulation,
        dist_matrix: torch.Tensor,
        waste: torch.Tensor,
        capacity: torch.Tensor,
        tau: torch.Tensor,
        eta: torch.Tensor,
    ) -> None:
        """Standard VPL substitution using ACO for worst teams.

        Args:
            pop_manager: Controller for individuals.
            dist_matrix: Global distances of shape [B, N, N].
            waste: Node fill levels of shape [B, N].
            capacity: Fleet capacity of shape [B].
            tau: Pheromone matrix of shape [B, N, N].
            eta: Heuristic visibility of shape [B, N, N].
        """
        batch_size = dist_matrix.shape[0]
        n_sub = int(self.n_teams * self.sub_rate)
        if n_sub <= 0:
            return

        # Identify worst indices by biased fitness (quality + diversity)
        worst_indices = pop_manager.biased_fitness.argsort(dim=1, descending=True)[:, :n_sub]

        # New constructions
        sub_tours = self._aco_construct(dist_matrix, tau, eta, n_sub)

        # Evaluate new constructions
        flat_sub = sub_tours.view(batch_size * n_sub, -1)
        expanded_dist = dist_matrix.repeat_interleave(n_sub, dim=0)
        expanded_waste = waste.repeat_interleave(n_sub, dim=0)
        expanded_capacity = capacity.repeat_interleave(n_sub, dim=0)

        _, sub_costs = vectorized_linear_split(flat_sub, expanded_dist, expanded_waste, expanded_capacity)

        for b in range(batch_size):
            for s in range(n_sub):
                idx = int(worst_indices[b, s].item())
                pop_manager.population[b, idx] = sub_tours[b, s]
                pop_manager.costs[b, idx] = sub_costs.view(batch_size, n_sub)[b, s]

        pop_manager.compute_biased_fitness(self.elite_size)

    def _routes_to_giant_tours(
        self,
        routes_list: Union[List[List[int]], List[torch.Tensor]],
        batch_size: int,
        n_teams: int,
        N: int,
        backup_giant: torch.Tensor,
    ) -> torch.Tensor:
        """Utility to convert coached route lists back to giant tours.

        Args:
            routes_list: Nested list or collection of routes.
            batch_size: Number of problem instances.
            n_teams: Population size (teams).
            N: Number of customers (total nodes - 1).
            backup_giant: Baseline tours for fallback of shape [B, K, N].

        Returns:
            torch.Tensor: Reshaped giant tours of shape [B, K, N].
        """
        device = backup_giant.device
        giant_tours = torch.zeros((batch_size, n_teams, N), dtype=torch.long, device=device)

        for b in range(batch_size):
            for t in range(n_teams):
                idx = b * n_teams + t
                routes = routes_list[idx]
                if isinstance(routes, torch.Tensor):
                    nodes = routes[routes != 0]
                else:
                    nodes = torch.tensor([n for n in routes if n != 0], device=device)

                if nodes.size(0) < N:
                    nodes = torch.cat(
                        [
                            nodes,
                            torch.zeros(N - nodes.size(0), dtype=torch.long, device=device),
                        ]
                    )
                elif nodes.size(0) > N:
                    nodes = nodes[:N]

                if nodes.sum() == 0:
                    giant_tours[b, t] = backup_giant[b, t].clone()
                else:
                    giant_tours[b, t] = nodes
        return giant_tours
