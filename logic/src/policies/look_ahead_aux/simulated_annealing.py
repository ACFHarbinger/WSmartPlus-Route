import copy
import math
import random
import time

from logic.src.policies.look_ahead_aux.computations import compute_total_cost
from logic.src.policies.look_ahead_aux.routes import uncross_arcs_in_sans_routes
from logic.src.policies.look_ahead_aux.sans_opt import (
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    cross_exchange,
    get_2opt_neighbors,
    insert_bin_in_route,
    move_between_routes,
    move_n_route_consecutive,
    move_n_route_random,
    mutate_route_by_swapping_bins,
    or_opt_move,
    relocate_within_route,
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)


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

    Args:
        routes (List[List[int]]): Initial routes.
        distance_matrix (np.ndarray): Distances.
        time_limit (float): Max runtime.
        id_to_index (Dict): Node ID mapping.
        data (pd.DataFrame): Dataset.
        vehicle_capacity (float): Vehicle limit.
        T_init (float): Start temperature.
        T_min (float): Stop temperature.
        alpha (float): Cooling rate.
        iterations_per_T (int): Steps at each temperature level.
        R (float): Revenue factor.
        V (float): Bin volume.
        density (float): Waste density.
        C (float): Distance cost factor.
        must_go_bins (List): Invariant nodes.
        removed_bins (List): Uncollected nodes.
        verbose (bool): Log progress.
        perc_bins_can_overflow (float): Threshold.
        volume (float): Bin size.
        density_val (float): Density.
        max_vehicles (int): Hard fleet limit.

    Returns:
        Tuple: (best_routes, best_removed_bins, stats_history).
    """

    def _power_function_decay(T_init, i, T_param):
        return T_init / (i**T_param)

    start_time = time.time()

    # --- 1. ROBUST INITIALIZATION ---
    if removed_bins is None:
        removed_bins = set()
    else:
        removed_bins = set(removed_bins)

    if must_go_bins is None:
        must_go_bins = set()
    else:
        must_go_bins = set(must_go_bins)

    # Capture any bins missed by the initial greedy solution
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
    current_solution = uncross_arcs_in_sans_routes(initial_solution, id_to_index, distance_matrix)

    # Calculate initial metrics
    current_cost = compute_total_cost(current_solution, distance_matrix, id_to_index)
    # Calculate initial metrics
    current_cost = compute_total_cost(current_solution, distance_matrix, id_to_index)

    # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
    real_kg = 0
    collected_must_go = set()
    if len(current_solution) > 0 and len(current_solution[0]) > 2:
        route0 = current_solution[0]
        stocks = dict(zip(data["#bin"], data["Stock"]))
        current_load = 0
        for b in route0:
            if b == 0:
                continue
            bin_kg = stocks.get(b, 0) * V * density / 100.0

            if current_load + bin_kg <= vehicle_capacity:
                current_load += bin_kg
                real_kg += bin_kg
                if must_go_bins and b in must_go_bins:
                    collected_must_go.add(b)
            else:
                break

    current_revenue = real_kg * R

    # Must-Go Penalty: Force all must_go_bins into the valid trunk of Route 0
    missed_must_go = len(must_go_bins) - len(collected_must_go) if must_go_bins else 0
    penalty_must_go = missed_must_go * 10000.0  # Huge penalty

    current_profit = current_revenue - current_cost - penalty_must_go

    initial_receita = current_revenue
    best_solution = copy.deepcopy(current_solution)
    best_profit = current_profit
    last_receita = initial_receita
    last_weight = real_kg
    last_distance = current_cost

    no_improvement_count = 0
    T = T_init
    stocks = dict(zip(data["#bin"], data["Stock"]))

    iter_count = 0
    while T > T_min:
        if time.time() - start_time > time_limit:
            if verbose:
                print("[DEBUG] Time limit reached.")
            break

        for _ in range(iterations_per_T):
            iter_count += 1
            if time.time() - start_time > time_limit:
                break

            # --- 2. STATE COPYING (Fixes Missing Bin 100) ---
            new_solution = copy.deepcopy(current_solution)
            candidate_removed_bins = copy.deepcopy(removed_bins)

            # --- 3. GUARD CLAUSES (Fixes IndexError Crash) ---
            # Separate operators into those needing routes and those needing removed bins
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

            # Smart Selection: If no routes, MUST add. If no removed bins, MUST modify routes.
            valid_ops = []
            if len(new_solution) > 0:
                valid_ops.extend(route_ops)
            if len(candidate_removed_bins) > 0:
                valid_ops.extend(add_ops)

            if not valid_ops:
                # Dead end (no routes and no bins to add), break or continue
                continue

            op = random.choice(valid_ops)

            # --- 4. APPLY OPERATORS ---
            # (Note: We pass candidate_removed_bins to operators, protecting the main set)
            if op == "2opt":
                # Guard: Route must be long enough
                valid_indices = [i for i, r in enumerate(new_solution) if len(r) > 3]
                if valid_indices:
                    r = random.choice(valid_indices)
                    new_solution[r] = random.choice(get_2opt_neighbors(new_solution[r]))
            elif op == "move":
                neighbors = move_between_routes(new_solution, data, vehicle_capacity, id_to_index)
                if neighbors:
                    new_solution = random.choice(neighbors)
            elif op == "swap":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = mutate_route_by_swapping_bins(new_solution[r], num_bins=random.choice([1, 2]))
            elif op == "remove":
                # Assuming remove_n_bins_random handles n=1 internally or we call specific func
                new_solution = remove_n_bins_random(new_solution, candidate_removed_bins, must_go_bins, n=1)
            elif op == "move_n_random":
                new_solution = move_n_route_random(new_solution, n=random.randint(2, 5))
            elif op == "move_n_consec":
                new_solution = move_n_route_consecutive(new_solution, n=random.randint(2, 5))
            elif op == "swap_n_random":
                new_solution = swap_n_route_random(new_solution, n=random.randint(2, 5))
            elif op == "swap_n_consec":
                new_solution = swap_n_route_consecutive(new_solution, n=random.randint(2, 5))
            elif op == "remove_n_bins":
                new_solution = remove_n_bins_random(
                    new_solution,
                    candidate_removed_bins,
                    must_go_bins,
                    n=random.randint(2, 5),
                )
            elif op == "remove_n_bins_consec":
                new_solution = remove_n_bins_consecutive(
                    new_solution,
                    candidate_removed_bins,
                    must_go_bins,
                    n=random.randint(2, 5),
                )
            elif op == "add_n_bins":
                new_solution = add_n_bins_random(
                    new_solution,
                    candidate_removed_bins,
                    stocks,
                    vehicle_capacity,
                    id_to_index,
                    distance_matrix,
                    n=2,
                )
            elif op == "add_n_bins_consec":
                new_solution = add_n_bins_consecutive(
                    new_solution,
                    candidate_removed_bins,
                    stocks,
                    vehicle_capacity,
                    id_to_index,
                    distance_matrix,
                    n=2,
                )
            elif op == "add_route_removed":
                new_solution = add_route_with_removed_bins_random(
                    new_solution, candidate_removed_bins, stocks, vehicle_capacity
                )
            elif op == "add_route_removed_consec":
                new_solution = add_route_with_removed_bins_consecutive(
                    new_solution, candidate_removed_bins, stocks, vehicle_capacity
                )
            elif op == "relocate":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = relocate_within_route(new_solution[r])
            elif op == "cross":
                new_solution = cross_exchange(new_solution)
            elif op == "or-opt":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = or_opt_move(new_solution[r])
            elif op == "insert":
                if not new_solution:
                    continue
                r = random.choice(range(len(new_solution)))
                all_bins = set(data["#bin"]) - {0}
                used_bins = set(b for route in new_solution for b in route)
                unused = list(all_bins - used_bins)
                if unused:
                    bin_to_insert = random.choice(unused)
                    load = sum(stocks.get(b, 0) for b in new_solution[r] if b != 0)
                    if load + stocks.get(bin_to_insert, 0) <= vehicle_capacity:
                        new_solution[r] = insert_bin_in_route(
                            new_solution[r], bin_to_insert, id_to_index, distance_matrix
                        )
                        # Remove from candidate removed bins if it was there
                        if bin_to_insert in candidate_removed_bins:
                            candidate_removed_bins.remove(bin_to_insert)

            # Filter out empty routes (safely)
            new_solution = [r for r in new_solution if len(r) > 2]

            # --- 5. EVALUATION ---
            # --- 5. EVALUATION ---
            new_cost = compute_total_cost(new_solution, distance_matrix, id_to_index)
            # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
            real_kg = 0
            collected_must_go = set()
            if len(new_solution) > 0 and len(new_solution[0]) > 2:
                route0 = new_solution[0]
                current_load = 0
                for b in route0:
                    if b == 0:
                        continue
                    # Optimization: pre-calculate bin weights? Or just do it here.
                    bin_kg = stocks.get(b, 0) * V * density / 100.0
                    if current_load + bin_kg <= vehicle_capacity:
                        current_load += bin_kg
                        real_kg += bin_kg
                        if must_go_bins and b in must_go_bins:
                            collected_must_go.add(b)
                    else:
                        break

            new_revenue = real_kg * R
            # Must-Go Penalty
            missed_must_go = len(must_go_bins) - len(collected_must_go) if must_go_bins else 0
            penalty_must_go = missed_must_go * 10000.0

            new_profit = new_revenue - new_cost - penalty_must_go

            delta = new_profit - current_profit

            # Acceptance Probability
            if delta > 0:
                accept = True
            else:
                p = math.exp(delta / T)
                accept = random.random() < p

            if accept:
                # ACCEPT: Adopt new routes AND the modified removed_bins set
                current_solution = new_solution
                current_cost = new_cost
                current_profit = new_profit
                removed_bins = candidate_removed_bins  # <--- CRITICAL: Update state only on acceptance

                if current_profit > best_profit:
                    best_solution = copy.deepcopy(current_solution)
                    best_profit = current_profit
                    last_weight = real_kg
                    last_distance = new_cost
                    last_receita = new_revenue
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # REJECT: Do nothing.
                # candidate_removed_bins is discarded automatically.
                # removed_bins remains untouched, so "lost" bins are effectively restored.
                no_improvement_count += 1

            T = T * alpha
            if verbose:
                print(f"Temperature cooled to {T:.4f}")

        if no_improvement_count > 500:
            if verbose:
                print("[INFO] Reaquecendo temperatura.")
            T = T_init  # start again from the original temperature
            no_improvement_count = 0

    best_solution = [r for r in best_solution if len(r) > 2]
    best_solution = uncross_arcs_in_sans_routes(best_solution, id_to_index, distance_matrix)
    return best_solution, best_profit, last_distance, last_weight, last_receita
