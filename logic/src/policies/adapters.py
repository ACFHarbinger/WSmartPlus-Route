"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Architecture:
-------------
- **PolicyAdapter**: Abstract base class defining execute() interface
- **Concrete Adapters**: Policy-specific implementations
  - RegularPolicyAdapter: Fixed-schedule periodic collection
  - LastMinutePolicyAdapter: Reactive threshold-based collection
  - NeuralPolicyAdapter: Deep RL models (Attention Models, GCNs)
  - VRPPPolicyAdapter: Prize-Collecting VRP (Gurobi/Hexaly)
  - LookAheadPolicyAdapter: Rolling-horizon optimization

- **PolicyFactory**: Factory method for selecting appropriate adapter

Benefits:
---------
1. Decouples simulator from policy implementations
2. Enables runtime policy switching
3. Standardizes parameter passing and result format
4. Simplifies adding new policies

Usage:
------
    adapter = PolicyFactory.get_adapter("am_gat")
    tour, cost, output = adapter.execute(**context)
"""
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Tuple, List
from logic.src.policies.neural_agent import NeuralAgent
from logic.src.utils.functions import move_to
from logic.src.policies import (
    local_search_2opt,
    get_route_cost, find_route, create_points, find_solutions,
    policy_lookahead, policy_lookahead_sans, policy_lookahead_vrpp,
    policy_lookahead_alns, policy_lookahead_hgs, policy_lookahead_bcp,
    policy_last_minute, policy_last_minute_and_path, policy_regular,
    policy_vrpp
)
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params


class PolicyAdapter(ABC):
    """
    Abstract base class for policy adapters.

    All policy adapters must implement the execute() method which takes
    a context dictionary and returns (tour, cost, additional_output).
    """
    @abstractmethod
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        pass

class RegularPolicyAdapter(PolicyAdapter):
    """
    Adapter for regular (periodic) collection policy.

    Executes fixed-schedule collection where all bins are visited every N days.
    Supports route caching for efficiency and both single/multi-vehicle routing.
    """
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        policy = kwargs['policy']
        bins = kwargs['bins']
        distancesC = kwargs['distancesC']
        day = kwargs['day']
        cached = kwargs['cached']
        waste_type = kwargs['waste_type']
        area = kwargs['area']
        n_vehicles = kwargs['n_vehicles']
        coords = kwargs['coords']
        distance_matrix = kwargs['distance_matrix']
        two_opt_max_iter = kwargs.get('two_opt_max_iter', 0)
        config = kwargs.get('config', {})
        regular_config = config.get('regular', {})

        lvl = int(policy.rsplit("_regular", 1)[1]) - 1
        
        # Override from config if present
        if 'level' in regular_config:
            lvl = int(regular_config['level']) - 1
            
        if lvl < 0: raise ValueError(f'Invalid lvl value for policy_regular: {lvl + 1}')
        tour = policy_regular(bins.n, bins.c, distancesC, lvl, day, cached, waste_type, area, n_vehicles, coords)
        if two_opt_max_iter > 0: tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
        cost = get_route_cost(distance_matrix, tour) if tour else 0
        if cached is not None and not cached and tour: cached = tour
        return tour, cost, cached

class LastMinutePolicyAdapter(PolicyAdapter):
    """
    Adapter for last-minute (reactive) collection policy.

    Executes threshold-based collection with optional path-based opportunistic
    collection. Only routes when bins exceed configured fill thresholds.
    """
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        policy = kwargs['policy']
        bins = kwargs['bins']
        distancesC = kwargs['distancesC']
        waste_type = kwargs['waste_type']
        area = kwargs['area']
        n_vehicles = kwargs['n_vehicles']
        coords = kwargs['coords']
        distance_matrix = kwargs['distance_matrix']
        two_opt_max_iter = kwargs.get('two_opt_max_iter', 0)
        paths_between_states = kwargs.get('paths_between_states')
        config = kwargs.get('config', {})
        last_minute_config = config.get('last_minute', {})

        if 'policy_last_minute_and_path' in policy:
            cf = int(policy.rsplit("_and_path", 1)[1])
            
            # Override from config if present
            if 'cf' in last_minute_config:
                cf = int(last_minute_config['cf'])
                
            if cf <= 0: raise ValueError(f'Invalid cf value for policy_last_minute_and_path: {cf}')
            bins.setCollectionLvlandFreq(cf=cf/100)
            tour = policy_last_minute_and_path(bins.c, distancesC, paths_between_states, bins.collectlevl, waste_type, area, n_vehicles, coords)
        else:
            cf = int(policy.rsplit("_last_minute", 1)[1])
            
            # Override from config if present
            if 'cf' in last_minute_config:
                cf = int(last_minute_config['cf'])
                
            if cf <= 0: raise ValueError(f'Invalid cf value for policy_last_minute: {cf}')
            bins.setCollectionLvlandFreq(cf=cf/100)
            tour = policy_last_minute(bins.c, distancesC, bins.collectlevl, waste_type, area, n_vehicles, coords)
        
        if two_opt_max_iter > 0: tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
        cost = get_route_cost(distance_matrix, tour) if tour else 0
        return tour, cost, None

class NeuralPolicyAdapter(PolicyAdapter):
    """
    Adapter for neural network-based policies.

    Executes deep reinforcement learning models (Attention Models, GCN-based)
    with optional HRL manager integration for gating and masking decisions.
    """
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        model_env = kwargs['model_env']
        model_ls = kwargs['model_ls']
        bins = kwargs['bins']
        device = kwargs['device']
        fill = kwargs['fill']
        dm_tensor = kwargs['dm_tensor']
        run_tsp = kwargs['run_tsp']
        hrl_manager = kwargs.get('hrl_manager')
        gate_prob_threshold = kwargs.get('gate_prob_threshold', 0.5)
        mask_prob_threshold = kwargs.get('mask_prob_threshold', 0.5)
        two_opt_max_iter = kwargs.get('two_opt_max_iter', 0)

        agent = NeuralAgent(model_env)
        model_data, graph, profit_vars = model_ls
        
        # set_daily_waste logic - duplicated here or helper?
        # Ideally helper. but for now implementing inline or reuse method if passed?
        # Or duplicating since it's small.
        model_data['waste'] = torch.as_tensor(bins.c, dtype=torch.float32).unsqueeze(0)/100.
        if 'fill_history' in model_data: 
            model_data['current_fill'] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)/100.
        daily_data = move_to(model_data, device)
        
        tour, cost, output_dict = agent.compute_simulator_day(
            daily_data, graph, dm_tensor, profit_vars, run_tsp, hrl_manager=hrl_manager, 
            waste_history=bins.get_level_history(device=device), threshold=gate_prob_threshold, 
            mask_threshold=mask_prob_threshold, two_opt_max_iter=two_opt_max_iter
        )
        return tour, cost, output_dict

class VRPPPolicyAdapter(PolicyAdapter):
    """
    Adapter for VRPP (Vehicle Routing Problem with Profits) policy.

    Executes Prize-Collecting VRP using Gurobi or Hexaly solvers.
    Optimizes profit while deciding which bins to collect.
    """
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        policy = kwargs['policy']
        bins = kwargs['bins']
        distance_matrix = kwargs['distance_matrix']
        model_env = kwargs['model_env']
        waste_type = kwargs['waste_type']
        area = kwargs['area']
        n_vehicles = kwargs['n_vehicles']
        distancesC = kwargs['distancesC']
        run_tsp = kwargs['run_tsp']
        two_opt_max_iter = kwargs.get('two_opt_max_iter', 0)
        config = kwargs.get('config', {})
        
        vrpp_config = config.get('vrpp', {})

        routes, _, _ = policy_vrpp(
            policy, bins.c, bins.means, bins.std, distance_matrix.tolist(),
            model_env, waste_type, area, n_vehicles, config=vrpp_config
        )
        tour = []
        cost = 0
        if routes:
            tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
            if two_opt_max_iter > 0: tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
            cost = get_route_cost(distance_matrix, tour)
        return tour, cost, None

class LookAheadPolicyAdapter(PolicyAdapter):
    """
    Adapter for look-ahead (rolling-horizon) policies.

    Executes optimization over a future time window to decide collections.
    Supports multiple solver backends: VRPP, SANS, HGS, ALNS, BCP, or OR solvers.
    """
    def execute(self, **kwargs) -> Tuple[List[int], float, Any]:
        policy = kwargs['policy']
        graph_size = kwargs['graph_size']
        bins = kwargs['bins']
        new_data = kwargs['new_data']
        coords = kwargs['coords']
        current_collection_day = kwargs['current_collection_day']
        area = kwargs['area']
        waste_type = kwargs['waste_type']
        n_vehicles = kwargs['n_vehicles']
        model_env = kwargs['model_env']
        distance_matrix = kwargs['distance_matrix']
        distancesC = kwargs['distancesC']
        run_tsp = kwargs['run_tsp']
        two_opt_max_iter = kwargs.get('two_opt_max_iter', 0)

        last_minute_config = kwargs.get('config', {}).get('lookahead', {})

        look_ahead_config = policy[policy.find('ahead_') + len('ahead_')]
        possible_configurations = {
            'a': [500,75,0.95,0,0.095,0,0], 
            'b': [2000,75,0.7,0,0.095,0,0]
        }
        
        # Override from config if present
        if look_ahead_config in last_minute_config:
            possible_configurations[look_ahead_config] = last_minute_config[look_ahead_config]
        
        try:
            chosen_combination = possible_configurations[look_ahead_config]
        except KeyError:
            print('Possible policy_look_ahead configurations:')
            for pos_pol, configs in possible_configurations.items():
                print(f'{pos_pol} configuration: {configs}')
            raise ValueError(f'Invalid policy_look_ahead configuration: {policy}')

        binsids = np.arange(0, graph_size).tolist()
        must_go_bins = policy_lookahead(binsids, bins.c, bins.means, current_collection_day)
        
        tour = []
        cost = 0
        if len(must_go_bins) > 0:
            vehicle_capacity, R, B, C, E = load_area_and_waste_type_params(area, waste_type)
            values = {
                'R': R, 'C': C, 'E': E, 'B': B, 
                'vehicle_capacity': vehicle_capacity,
                'Omega': last_minute_config.get('Omega', 0.1),
                'delta': last_minute_config.get('delta', 0),  
                'psi': last_minute_config.get('psi', 1),
            }
            routes = None
            if 'vrpp' in policy:
                # VRPP specific lookahead config? Usually handled by generic values but let's be safe
                vrpp_la_config = last_minute_config.get('vrpp', {})
                # values.update(vrpp_la_config) # If we want full override
                
                routes, _, _ = policy_lookahead_vrpp(
                    bins.c, binsids, must_go_bins, distance_matrix, values, 
                    number_vehicles=n_vehicles, env=model_env, time_limit=vrpp_la_config.get('time_limit', 60)
                )
            elif 'sans' in policy:
                sans_config = last_minute_config.get('sans', {})
                values['time_limit'] = sans_config.get('time_limit', 60)
                values['perc_bins_can_overflow'] = sans_config.get('perc_bins_can_overflow', 0)
                
                T_min = sans_config.get('T_min', 0.01)
                T_init = sans_config.get('T_init', 75)
                iterations_per_T = sans_config.get('iterations_per_T', 5000)
                alpha = sans_config.get('alpha', 0.95)
                
                params = (T_init, iterations_per_T, alpha, T_min)
                new_data.loc[1:, 'Stock'] = bins.c.astype('float32')
                new_data.loc[1:, 'Accum_Rate'] = bins.means.astype('float32')
                routes, _, _ = policy_lookahead_sans(new_data, coords, distance_matrix, params, must_go_bins, values)
                if routes: routes = routes[0]
            elif 'hgs' in policy:
                hgs_config = last_minute_config.get('hgs', {})
                values['time_limit'] = hgs_config.get('time_limit', 60)
                routes, _, _ = policy_lookahead_hgs(bins.c, binsids, must_go_bins, distance_matrix, values, coords)
            elif 'alns' in policy:
                alns_config = last_minute_config.get('alns', {})
                values['time_limit'] = alns_config.get('time_limit', 60)
                values['Iterations'] = alns_config.get('Iterations', 5000)
                variant = alns_config.get('variant', 'default')
                # Overwrite if encoded in name? Or trust config?
                if 'package' in policy: variant = 'package'
                elif 'ortools' in policy: variant = 'ortools'
                # If config specifies something else, maybe config wins? user request implies config is master.
                if alns_config.get('variant'): variant = alns_config.get('variant')
                
                routes, _, _ = policy_lookahead_alns(bins.c, binsids, must_go_bins, distance_matrix, values, coords, variant=variant)
            elif 'bcp' in policy:
                bcp_config = last_minute_config.get('bcp', {})
                values['time_limit'] = bcp_config.get('time_limit', 60)
                values['Iterations'] = bcp_config.get('Iterations', 50)
                routes, _, _ = policy_lookahead_bcp(bins.c, binsids, must_go_bins, distance_matrix, values, coords)
            else:
                values['shift_duration'] = 390
                values['perc_bins_can_overflow'] = 0
                points = create_points(new_data, coords)
                new_data.loc[1:graph_size+1, 'Stock'] = (bins.c/100).astype('float32')
                new_data.loc[1:graph_size+1, 'Accum_Rate'] = (bins.means/100).astype('float32')
                try:
                    routes, _, _ = find_solutions(new_data, coords, distance_matrix, chosen_combination, must_go_bins, values, graph_size, points, time_limit=600)
                except:
                    routes, _, _ = find_solutions(new_data, coords, distance_matrix, chosen_combination, must_go_bins, values, graph_size, points, time_limit=3600)
                if routes: routes = routes[0]

            if routes:
                tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
                if two_opt_max_iter > 0: tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
                cost = get_route_cost(distance_matrix, tour)
        else:
            tour = [0, 0]
            cost = 0
        return tour, cost, None


class PolicyFactory:
    """
    Factory for creating policy adapters.

    Implements the Factory Method pattern to instantiate the appropriate
    PolicyAdapter based on the policy name string.

    Policy Name Patterns:
    ---------------------
    - 'policy_regular*': RegularPolicyAdapter
    - 'policy_last_minute*': LastMinutePolicyAdapter
    - 'am*', 'ddam*', 'transgcn*': NeuralPolicyAdapter
    - '*vrpp*' with 'gurobi' or 'hexaly': VRPPPolicyAdapter
    - 'policy_look_ahead*': LookAheadPolicyAdapter
    """
    @staticmethod
    def get_adapter(policy_name: str) -> PolicyAdapter:
        """
        Create and return the appropriate PolicyAdapter for the given policy name.

        Args:
            policy_name (str): Policy identifier string

        Returns:
            PolicyAdapter: Concrete adapter instance

        Raises:
            ValueError: If policy name doesn't match any known pattern
        """
        if 'policy_last_minute' in policy_name:
            return LastMinutePolicyAdapter()
        elif 'policy_regular' in policy_name:
            return RegularPolicyAdapter()
        elif policy_name[:2] == 'am' or policy_name[:4] == 'ddam' or "transgcn" in policy_name:
            return NeuralPolicyAdapter()
        elif ('gurobi' in policy_name or 'hexaly' in policy_name) and 'vrpp' in policy_name:
            return VRPPPolicyAdapter()
        elif 'policy_look_ahead' in policy_name:
            return LookAheadPolicyAdapter()
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
