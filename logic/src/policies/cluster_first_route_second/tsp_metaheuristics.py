"""
TSP Metaheuristic Solvers for Cluster-First Route-Second Algorithm.

This module provides metaheuristic algorithms for solving the Traveling Salesman Problem
in the routing phase of the VFJ algorithm.

Supported Metaheuristics (as evaluated in Sultana & Akhand, 2017):
- Particle Swarm Optimization (PSO) - FULLY IMPLEMENTED
- Ant Colony Optimization (ACO) - Placeholder
- Genetic Algorithm (GA) - Placeholder

The paper concludes that VFJ + PSO provides the best performance for CVRP benchmarks.

Reference:
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
    Section 4: "VFJ algorithm with PSO performs better than VFJ algorithm with
    ACO and VFJ algorithm with GA"
"""

import time
from typing import List, Tuple

import numpy as np


def _calculate_tour_distance(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a TSP tour."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i], tour[i + 1]]
    return total


def _two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """
    Perform 2-opt swap on tour segment between indices i and k.

    2-opt removes two edges and reconnects the path by reversing the segment.
    This is the fundamental local search operator for TSP.

    Args:
        tour: Current tour (includes depot at start and end)
        i: Start index of segment to reverse
        k: End index of segment to reverse

    Returns:
        New tour with reversed segment
    """
    new_tour = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
    return new_tour


def _apply_swap_sequence(tour: List[int], swap_ops: List[Tuple[int, int]], max_swaps: int = 3) -> List[int]:
    """
    Apply a sequence of swap operations to move toward a target tour.

    This implements the discrete PSO velocity update mechanism by
    probabilistically applying swaps that move the current particle
    toward its personal best and the global best.

    Args:
        tour: Current tour
        swap_ops: List of (i, j) swap operations
        max_swaps: Maximum number of swaps to apply

    Returns:
        Modified tour
    """
    result = tour.copy()
    num_swaps = min(len(swap_ops), max_swaps)

    for i in range(num_swaps):
        idx1, idx2 = swap_ops[i]
        if 0 < idx1 < len(result) - 1 and 0 < idx2 < len(result) - 1:
            result[idx1], result[idx2] = result[idx2], result[idx1]

    return result


def _get_swap_sequence(source: List[int], target: List[int]) -> List[Tuple[int, int]]:
    """
    Compute sequence of swap operations to transform source tour into target tour.

    This is used to compute the "velocity" in discrete PSO by finding
    the swaps needed to move from current position to pbest/gbest.

    Args:
        source: Current tour
        target: Target tour

    Returns:
        List of (i, j) index pairs to swap
    """
    swaps = []
    temp = source.copy()

    for i in range(1, len(temp) - 1):  # Exclude depot at start and end
        if temp[i] != target[i]:
            # Find where target[i] is in temp
            for j in range(i + 1, len(temp) - 1):
                if temp[j] == target[i]:
                    swaps.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                    break

    return swaps


def find_route_pso(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    n_particles: int = 30,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> List[int]:
    """
    Solve TSP using Particle Swarm Optimization (PSO).

    PSO is a population-based stochastic optimization technique inspired by social
    behavior of bird flocking. For TSP, particles represent tour permutations and
    "velocity" is represented by swap sequences.

    Algorithm:
    1. Initialize swarm of random tours
    2. Evaluate fitness (tour distance)
    3. Update personal best (pbest) and global best (gbest)
    4. Update velocity: sequence of swaps toward pbest and gbest
    5. Apply velocity to particles with inertia, cognitive, and social components
    6. Repeat until time limit or convergence

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices to visit (including depot at index 0).
        time_limit: Maximum time in seconds for optimization.
        seed: Random seed for reproducibility.
        n_particles: Number of particles in swarm (default: 30).
        w: Inertia weight (default: 0.7) - controls exploration vs exploitation.
        c1: Cognitive coefficient (default: 1.5) - attraction to personal best.
        c2: Social coefficient (default: 1.5) - attraction to global best.

    Returns:
        TSP tour as list of node indices [0, i1, i2, ..., in, 0].

    Reference:
        Sultana et al. (2017), Section 3.3: "PSO for TSP routing phase"

    Implementation Notes:
        - Uses discrete PSO with swap sequences as velocity
        - Applies 2-opt local search to improve solutions
        - Respects time limit using time.time()
    """
    np.random.seed(seed)
    start_time = time.time()

    # Handle edge cases
    if not cluster:
        return [0, 0]

    # Ensure depot is in cluster
    if 0 not in cluster:
        cluster = [0] + cluster

    # Extract customer nodes (exclude depot)
    customers = [node for node in cluster if node != 0]
    n_customers = len(customers)

    # Handle trivial case
    if n_customers == 0:
        return [0, 0]
    if n_customers == 1:
        return [0, customers[0], 0]

    # Initialize swarm: each particle is a tour [0, perm, 0]
    particles = []
    velocities = []  # Velocities are swap sequences
    pbest_tours = []  # Personal best tours
    pbest_distances = []  # Personal best distances

    for _ in range(n_particles):
        # Random permutation of customers
        perm = np.random.permutation(customers).tolist()
        tour = [0] + perm + [0]
        particles.append(tour)
        velocities.append([])  # Initial velocity is empty

        # Personal best initialized to current position
        dist = _calculate_tour_distance(tour, distance_matrix)
        pbest_tours.append(tour.copy())
        pbest_distances.append(dist)

    # Find initial global best
    gbest_idx = np.argmin(pbest_distances)
    gbest_tour = pbest_tours[gbest_idx].copy()
    gbest_distance = pbest_distances[gbest_idx]

    # PSO main loop
    iteration = 0
    no_improvement_count = 0

    while time.time() - start_time < time_limit:
        iteration += 1
        improved = False

        for p_idx in range(n_particles):
            # Check time limit
            if time.time() - start_time >= time_limit:
                break

            # Get swap sequences toward pbest and gbest
            swaps_to_pbest = _get_swap_sequence(particles[p_idx], pbest_tours[p_idx])
            swaps_to_gbest = _get_swap_sequence(particles[p_idx], gbest_tour)

            # Update velocity with inertia, cognitive, and social components
            new_velocity = []

            # Inertia: keep some of previous velocity
            if np.random.rand() < w and velocities[p_idx]:
                num_keep = min(len(velocities[p_idx]), 2)
                new_velocity.extend(velocities[p_idx][:num_keep])

            # Cognitive: move toward personal best
            if np.random.rand() < c1 and swaps_to_pbest:
                num_cognitive = max(1, int(len(swaps_to_pbest) * c1 / 3))
                new_velocity.extend(swaps_to_pbest[:num_cognitive])

            # Social: move toward global best
            if np.random.rand() < c2 and swaps_to_gbest:
                num_social = max(1, int(len(swaps_to_gbest) * c2 / 3))
                new_velocity.extend(swaps_to_gbest[:num_social])

            velocities[p_idx] = new_velocity

            # Apply velocity to particle
            if new_velocity:
                particles[p_idx] = _apply_swap_sequence(particles[p_idx], new_velocity, max_swaps=3)

            # Apply 2-opt local search with probability
            if np.random.rand() < 0.3 and n_customers >= 4:
                best_tour = particles[p_idx].copy()
                best_dist = _calculate_tour_distance(best_tour, distance_matrix)

                # Try a limited number of 2-opt swaps
                for _ in range(min(10, n_customers)):
                    i = np.random.randint(1, n_customers)
                    k = np.random.randint(i + 1, n_customers + 1)
                    new_tour = _two_opt_swap(best_tour, i, k)
                    new_dist = _calculate_tour_distance(new_tour, distance_matrix)

                    if new_dist < best_dist:
                        best_tour = new_tour
                        best_dist = new_dist

                particles[p_idx] = best_tour

            # Evaluate particle fitness
            current_dist = _calculate_tour_distance(particles[p_idx], distance_matrix)

            # Update personal best
            if current_dist < pbest_distances[p_idx]:
                pbest_tours[p_idx] = particles[p_idx].copy()
                pbest_distances[p_idx] = current_dist

                # Update global best
                if current_dist < gbest_distance:
                    gbest_tour = particles[p_idx].copy()
                    gbest_distance = current_dist
                    improved = True
                    no_improvement_count = 0

        # Early stopping if no improvement
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 20:
                break

    return gbest_tour


def find_route_aco(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
) -> List[int]:
    """
    Solve TSP using Ant Colony Optimization (ACO).

    ACO is a probabilistic technique inspired by the foraging behavior of ants.
    Artificial ants construct solutions by probabilistically choosing edges based on
    pheromone trails and heuristic information (inverse distance).

    PLACEHOLDER IMPLEMENTATION:
    This is currently a stub that falls back to nearest neighbor heuristic.
    For production use, integrate an ACO library such as:
    - ACOpy (https://github.com/Akavall/ACOpy)
    - scikit-opt (https://github.com/guofei9987/scikit-opt)
    - Python-TSP (https://github.com/fillipe-gsm/python-tsp)

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices to visit (including depot at index 0).
        time_limit: Maximum time in seconds for optimization.
        seed: Random seed for reproducibility.

    Returns:
        TSP tour as list of node indices [0, i1, i2, ..., in, 0].

    Reference:
        Sultana et al. (2017), Section 3.3: ACO for TSP routing phase.

    TODO: Replace with actual ACO implementation.
    """
    # Placeholder: Use nearest neighbor heuristic
    # In production, replace with ACO algorithm
    np.random.seed(seed)

    if not cluster:
        return [0, 0]

    # Ensure depot is at start and end
    if 0 not in cluster:
        cluster = [0] + cluster

    # Nearest neighbor heuristic (placeholder for ACO)
    unvisited = set(cluster) - {0}
    tour = [0]
    current = 0

    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    tour.append(0)  # Return to depot
    return tour


def find_route_ga(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
) -> List[int]:
    """
    Solve TSP using Genetic Algorithm (GA).

    GA is an evolutionary algorithm that evolves a population of candidate solutions
    through selection, crossover, and mutation operators. For TSP, specialized
    crossover operators like PMX or OX preserve tour validity.

    PLACEHOLDER IMPLEMENTATION:
    This is currently a stub that falls back to nearest neighbor heuristic.
    For production use, integrate a GA library or implement custom GA with:
    - Ordered crossover (OX) or partially mapped crossover (PMX)
    - Swap/inversion mutation
    - Tournament selection

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices to visit (including depot at index 0).
        time_limit: Maximum time in seconds for optimization.
        seed: Random seed for reproducibility.

    Returns:
        TSP tour as list of node indices [0, i1, i2, ..., in, 0].

    Reference:
        Sultana et al. (2017), Section 3.3: GA for TSP routing phase.

    TODO: Replace with actual GA implementation.
    """
    # Placeholder: Use nearest neighbor heuristic
    # In production, replace with GA algorithm
    np.random.seed(seed)

    if not cluster:
        return [0, 0]

    # Ensure depot is at start and end
    if 0 not in cluster:
        cluster = [0] + cluster

    # Nearest neighbor heuristic (placeholder for GA)
    unvisited = set(cluster) - {0}
    tour = [0]
    current = 0

    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    tour.append(0)  # Return to depot
    return tour
