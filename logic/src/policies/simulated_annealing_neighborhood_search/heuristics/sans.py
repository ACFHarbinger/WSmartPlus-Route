"""
Simulated Annealing Algorithm.
"""

import copy
import math
import random
import time

from ..common.routes import uncross_arcs_in_sans_routes
from .sans_operators import apply_operator
from .sans_state import compute_profit


def improved_simulated_annealing(
    routes,
    distance_matrix,
    time_limit,
    id_to_index,
    data,
    vehicle_capacity,
    T_init=1000,
    T_min=0.001,
    alpha=0.995,
    iterations_per_T=100,
    R=0.165,
    V=2.5,
    density=20,
    C=1.0,
    must_go_bins=None,
    removed_bins=None,
    verbose=False,
    perc_bins_can_overflow=0.0,
    volume=2.5,
    density_val=20,
    max_vehicles=None,
):
    """
    Refine routing solutions using a multi-neighborhood Simulated Annealing algorithm.
    """
    start_time = time.time()

    # --- 1. ROBUST INITIALIZATION ---
    # Prepare the initial solution and ensure state consistency between routes and removed bins.
    if removed_bins is None:
        removed_bins = set()
    else:
        removed_bins = set(removed_bins)

    if must_go_bins is None:
        must_go_bins = set()
    else:
        must_go_bins = set(must_go_bins)

    # State Sync: Capture any bins missed by the initial greedy solution.
    # Every bin ID must be either in a route or in the 'removed_bins' set.
    all_bins_ids = set(data["#bin"].tolist()) - {0}
    scheduled_bins = set()
    for r in routes:
        for b in r:
            if b != 0:
                scheduled_bins.add(b)
    missing_bins = all_bins_ids - scheduled_bins
    for b in missing_bins:
        removed_bins.add(b)

    initial_solution = copy.deepcopy(routes)
    # Post-processing: Uncross arcs in the initial solution to start from a geometric optimum.
    current_solution = uncross_arcs_in_sans_routes(initial_solution, id_to_index, distance_matrix)

    stocks = dict(zip(data["#bin"], data["Stock"]))

    # Initial Evaluation: Calculate the starting profit, cost, and revenue.
    current_profit, current_cost, current_revenue, real_kg = compute_profit(
        current_solution,
        distance_matrix,
        id_to_index,
        data,
        vehicle_capacity,
        R,
        V,
        density,
        must_go_bins,
        stocks,
    )

    initial_receita = current_revenue
    best_solution = copy.deepcopy(current_solution)
    best_profit = current_profit
    last_receita = initial_receita
    last_weight = real_kg
    last_distance = current_cost

    no_improvement_count = 0
    T = T_init

    iter_count = 0
    # --- 2. MAIN SIMULATED ANNEALING LOOP ---
    # The temperature T gradually decreases from T_init to T_min.
    while T > T_min:
        if time.time() - start_time > time_limit:
            if verbose:
                print("[DEBUG] Time limit reached.")
            break

        # Inner Loop: Perform multiple iterations at each temperature level.
        for _ in range(iterations_per_T):
            iter_count += 1
            if time.time() - start_time > time_limit:
                break

            # --- 3. STATE COPYING ---
            new_solution = copy.deepcopy(current_solution)
            candidate_removed_bins = copy.deepcopy(removed_bins)

            # --- 4. NEIGHBORHOOD MOVE SELECTION (SANS Strategy) ---
            # SANS uses a multi-neighborhood approach. We define two sets of operators:
            # - route_ops: Modify existing routes (intra/inter route moves).
            # - add_ops: Move bins between 'removed_bins' and the active 'routes'.
            route_ops = [
                "2opt",
                "move",
                "swap",
                "remove",
                "insert",
                "move_n_random",
                "move_n_consec",
                "swap_n_random",
                "swap_n_consec",
                "remove_n_bins",
                "remove_n_bins_consec",
                "relocate",
                "cross",
                "or-opt",
            ]

            add_ops = [
                "add_n_bins",
                "add_n_bins_consec",
                "add_route_removed",
                "add_route_removed_consec",
            ]

            # Smart Selection: Only choose operators that are valid for the current state.
            valid_ops = []
            if len(new_solution) > 0:
                valid_ops.extend(route_ops)
            if len(candidate_removed_bins) > 0:
                valid_ops.extend(add_ops)

            if not valid_ops:
                continue

            # Randomly select and apply one operator to generate a neighbor 'new_solution'.
            op = random.choice(valid_ops)

            # --- 5. APPLY OPERATOR ---
            new_solution = apply_operator(
                op,
                new_solution,
                candidate_removed_bins,
                data,
                vehicle_capacity,
                id_to_index,
                stocks,
                must_go_bins,
                distance_matrix,
            )

            # Filter out empty routes (safely)
            new_solution = [r for r in new_solution if len(r) > 2]

            # --- 6. EVALUATION ---
            # Compute the objective value (profit) of the neighbor solution.
            new_profit, new_cost, new_revenue, new_real_kg = compute_profit(
                new_solution,
                distance_matrix,
                id_to_index,
                data,
                vehicle_capacity,
                R,
                V,
                density,
                must_go_bins,
                stocks,
            )

            delta = new_profit - current_profit

            # --- 7. METROPOLIS ACCEPTANCE CRITERION ---
            # 1. Improvements (delta > 0) are always accepted.
            # 2. Deteriorations are accepted with probability exp(delta / T).
            # This allows the algorithm to escape local optima early in the search.
            if delta > 0:
                accept = True
            else:
                p = math.exp(delta / T)
                accept = random.random() < p

            if accept:
                # ACCEPT the new state
                current_solution = new_solution
                current_cost = new_cost
                current_profit = new_profit
                removed_bins = candidate_removed_bins

                # TRACK the overall best solution found so far
                if current_profit > best_profit:
                    best_solution = copy.deepcopy(current_solution)
                    best_profit = current_profit
                    last_weight = new_real_kg
                    last_distance = new_cost
                    last_receita = new_revenue
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # REJECT and keep the current state
                no_improvement_count += 1

        # --- 8. COOLING SCHEDULE ---
        # Reduce the temperature according to a geometric factor 'alpha'.
        # As T becomes small, the probability of accepting deteriorations approaches zero.
        T = T * alpha
        if verbose:
            print(f"Temperature cooled to {T:.4f}")

        # --- 9. REHEATING (Escaping Stagnation) ---
        # If no improvement is found for a long time, reset the temperature to T_init.
        # This implementation uses a simplified reheating strategy.
        if no_improvement_count > 500:
            if verbose:
                print("[INFO] Rehearing temperature.")
            T = T_init
            no_improvement_count = 0

    best_solution = [r for r in best_solution if len(r) > 2]
    best_solution = uncross_arcs_in_sans_routes(best_solution, id_to_index, distance_matrix)
    return best_solution, best_profit, last_distance, last_weight, last_receita
