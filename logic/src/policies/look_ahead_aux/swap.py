"""
Node exchange operators for solution perturbation and neighborhood search.

Provides intra-route and inter-route bin swapping logic. Includes routines
for swapping single points or entire sequences (segments) between routes,
supporting randomized and consecutive exchange strategies.
"""

from copy import deepcopy
from random import sample as rsample

__all__ = [
    "swap_1_route",
    "swap_2_routes",
    "swap_n_route_random",
    "swap_n_route_consecutive",
    "swap_n_2_routes_random",
    "swap_n_2_routes_consecutive",
]


# Function to swap in 1 route (Choose one route, pick two bins from the route and swap them)
def swap_1_route(routes_list):
    """
    Intra-route perturbation: Swap two random bins within the same route.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        if len(chosen_route) > 3:
            swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
            position_bin_1 = chosen_route.index(swap_bin_1)
            swap_bin_2 = swap_bin_1
            while swap_bin_2 == swap_bin_1:
                swap_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

            position_bin_2 = chosen_route.index(swap_bin_2)
            chosen_route[position_bin_1], chosen_route[position_bin_2] = (
                chosen_route[position_bin_2],
                chosen_route[position_bin_1],
            )


# Function to swap two bins between two different routes (Choose two routes and swap two bins between routes)
def swap_2_routes(routes_list):
    """
    Inter-route perturbation: Swap one bin from one route with another bin
    from a different route.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 1:
        chosen_route_1 = rsample(routes_list, 1)[0]
        chosen_route_2 = chosen_route_1
        while chosen_route_2 == chosen_route_1:
            chosen_route_2 = rsample(routes_list, 1)[0]

        bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]
        bin_1_position = chosen_route_1.index(bin_1)
        bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]
        bin_2_position = chosen_route_2.index(bin_2)
        chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
            chosen_route_2[bin_2_position],
            chosen_route_1[bin_1_position],
        )


# Function to swap n random bins inside a route
def swap_n_route_random(routes_list):
    """
    Intra-route perturbation: Swap n pairs of random bins within their routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    chosen_n = None
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        possible_n = [2, 3, 4, 5]
        chosen_n = rsample(possible_n, 1)[0]
        if len(chosen_route) > chosen_n + 2:
            if chosen_n == 2:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = swap_bin_1
                while swap_bin_2 == swap_bin_1:
                    swap_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_2 = chosen_route.index(swap_bin_2)
                chosen_route[position_bin_1], chosen_route[position_bin_2] = (
                    chosen_route[position_bin_2],
                    chosen_route[position_bin_1],
                )
                swap_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = swap_bin_3
                while swap_bin_4 == swap_bin_3:
                    swap_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_4 = chosen_route.index(swap_bin_4)
                chosen_route[position_bin_3], chosen_route[position_bin_4] = (
                    chosen_route[position_bin_4],
                    chosen_route[position_bin_3],
                )
            elif chosen_n == 3:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = swap_bin_1
                while swap_bin_2 == swap_bin_1:
                    swap_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_2 = chosen_route.index(swap_bin_2)
                chosen_route[position_bin_1], chosen_route[position_bin_2] = (
                    chosen_route[position_bin_2],
                    chosen_route[position_bin_1],
                )
                swap_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = swap_bin_3
                while swap_bin_4 == swap_bin_3:
                    swap_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_4 = chosen_route.index(swap_bin_4)
                chosen_route[position_bin_3], chosen_route[position_bin_4] = (
                    chosen_route[position_bin_4],
                    chosen_route[position_bin_3],
                )
                swap_bin_5 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_5 = chosen_route.index(swap_bin_5)
                swap_bin_6 = swap_bin_5
                while swap_bin_6 == swap_bin_5:
                    swap_bin_6 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_6 = chosen_route.index(swap_bin_6)
                chosen_route[position_bin_5], chosen_route[position_bin_6] = (
                    chosen_route[position_bin_6],
                    chosen_route[position_bin_5],
                )
            elif chosen_n == 4:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = swap_bin_1
                while swap_bin_2 == swap_bin_1:
                    swap_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_2 = chosen_route.index(swap_bin_2)
                chosen_route[position_bin_1], chosen_route[position_bin_2] = (
                    chosen_route[position_bin_2],
                    chosen_route[position_bin_1],
                )
                swap_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = swap_bin_3
                while swap_bin_4 == swap_bin_3:
                    swap_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_4 = chosen_route.index(swap_bin_4)
                chosen_route[position_bin_3], chosen_route[position_bin_4] = (
                    chosen_route[position_bin_4],
                    chosen_route[position_bin_3],
                )
                swap_bin_5 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_5 = chosen_route.index(swap_bin_5)
                swap_bin_6 = swap_bin_5
                while swap_bin_6 == swap_bin_5:
                    swap_bin_6 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_6 = chosen_route.index(swap_bin_6)
                chosen_route[position_bin_5], chosen_route[position_bin_6] = (
                    chosen_route[position_bin_6],
                    chosen_route[position_bin_5],
                )
                swap_bin_7 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_7 = chosen_route.index(swap_bin_7)
                swap_bin_8 = swap_bin_7
                while swap_bin_8 == swap_bin_7:
                    swap_bin_8 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_8 = chosen_route.index(swap_bin_8)
                chosen_route[position_bin_7], chosen_route[position_bin_8] = (
                    chosen_route[position_bin_8],
                    chosen_route[position_bin_7],
                )
            else:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = swap_bin_1
                while swap_bin_2 == swap_bin_1:
                    swap_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_2 = chosen_route.index(swap_bin_2)
                chosen_route[position_bin_1], chosen_route[position_bin_2] = (
                    chosen_route[position_bin_2],
                    chosen_route[position_bin_1],
                )
                swap_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = swap_bin_3
                while swap_bin_4 == swap_bin_3:
                    swap_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_4 = chosen_route.index(swap_bin_4)
                chosen_route[position_bin_3], chosen_route[position_bin_4] = (
                    chosen_route[position_bin_4],
                    chosen_route[position_bin_3],
                )
                swap_bin_5 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_5 = chosen_route.index(swap_bin_5)
                swap_bin_6 = swap_bin_5
                while swap_bin_6 == swap_bin_5:
                    swap_bin_6 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_6 = chosen_route.index(swap_bin_6)
                chosen_route[position_bin_5], chosen_route[position_bin_6] = (
                    chosen_route[position_bin_6],
                    chosen_route[position_bin_5],
                )
                swap_bin_7 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_7 = chosen_route.index(swap_bin_7)
                swap_bin_8 = swap_bin_7
                while swap_bin_8 == swap_bin_7:
                    swap_bin_8 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_8 = chosen_route.index(swap_bin_8)
                chosen_route[position_bin_7], chosen_route[position_bin_8] = (
                    chosen_route[position_bin_8],
                    chosen_route[position_bin_7],
                )
                swap_bin_9 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_9 = chosen_route.index(swap_bin_9)
                swap_bin_10 = swap_bin_9
                while swap_bin_10 == swap_bin_9:
                    swap_bin_10 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_bin_10 = chosen_route.index(swap_bin_10)
                chosen_route[position_bin_9], chosen_route[position_bin_10] = (
                    chosen_route[position_bin_10],
                    chosen_route[position_bin_9],
                )
    return chosen_n


# Function to swap n random bins inside a route
def swap_n_route_consecutive(routes_list):
    """
    Intra-route perturbation: Swap two consecutive sequences of n bins
    within the same route.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    chosen_n = None
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        possible_n = [2, 3, 4, 5]
        chosen_n = rsample(possible_n, 1)[0]
        if len(chosen_route) > chosen_n + 2:
            if chosen_n == 2:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 2], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = chosen_route[position_bin_1 + 1]
                position_bin_2 = chosen_route.index(swap_bin_2)
                chosen_route_copy = deepcopy(chosen_route)
                del chosen_route_copy[position_bin_1 - 1 : position_bin_1 + 2]
                if chosen_route_copy[0] != 0:
                    chosen_route_copy = [0] + chosen_route_copy

                if chosen_route_copy[-1] != 0:
                    chosen_route_copy = chosen_route_copy + [0]

                if len(chosen_route_copy) > chosen_n + 2:
                    swap_bin_3 = rsample(chosen_route_copy[1 : len(chosen_route_copy) - 2], 1)[0]
                    position_bin_3 = chosen_route.index(swap_bin_3)
                    chosen_route[position_bin_1], chosen_route[position_bin_3] = (
                        chosen_route[position_bin_3],
                        chosen_route[position_bin_1],
                    )
                    swap_bin_4 = chosen_route[position_bin_3 + 1]
                    position_bin_4 = chosen_route.index(swap_bin_4)
                    chosen_route[position_bin_2], chosen_route[position_bin_4] = (
                        chosen_route[position_bin_4],
                        chosen_route[position_bin_2],
                    )
            elif chosen_n == 3:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 3], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = chosen_route[position_bin_1 + 1]
                position_bin_2 = chosen_route.index(swap_bin_2)
                swap_bin_3 = chosen_route[position_bin_1 + 2]
                position_bin_3 = chosen_route.index(swap_bin_3)
                chosen_route_copy = deepcopy(chosen_route)
                del chosen_route_copy[position_bin_1 - 2 : position_bin_1 + 3]
                if chosen_route_copy[0] != 0:
                    chosen_route_copy = [0] + chosen_route_copy

                if chosen_route_copy[-1] != 0:
                    chosen_route_copy = chosen_route_copy + [0]

                if len(chosen_route_copy) > chosen_n + 2:
                    swap_bin_4 = rsample(chosen_route_copy[1 : len(chosen_route_copy) - 3], 1)[0]
                    position_bin_4 = chosen_route.index(swap_bin_4)
                    chosen_route[position_bin_1], chosen_route[position_bin_4] = (
                        chosen_route[position_bin_4],
                        chosen_route[position_bin_1],
                    )
                    swap_bin_5 = chosen_route[position_bin_4 + 1]
                    position_bin_5 = chosen_route.index(swap_bin_5)
                    chosen_route[position_bin_2], chosen_route[position_bin_5] = (
                        chosen_route[position_bin_5],
                        chosen_route[position_bin_2],
                    )
                    swap_bin_6 = chosen_route[position_bin_4 + 2]
                    position_bin_6 = chosen_route.index(swap_bin_6)
                    chosen_route[position_bin_3], chosen_route[position_bin_6] = (
                        chosen_route[position_bin_6],
                        chosen_route[position_bin_3],
                    )
            elif chosen_n == 4:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 4], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = chosen_route[position_bin_1 + 1]
                position_bin_2 = chosen_route.index(swap_bin_2)
                swap_bin_3 = chosen_route[position_bin_1 + 2]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = chosen_route[position_bin_1 + 3]
                position_bin_4 = chosen_route.index(swap_bin_4)
                chosen_route_copy = deepcopy(chosen_route)
                del chosen_route_copy[position_bin_1 - 3 : position_bin_1 + 4]

                if chosen_route_copy[0] != 0:
                    chosen_route_copy = [0] + chosen_route_copy

                if chosen_route_copy[-1] != 0:
                    chosen_route_copy = chosen_route_copy + [0]

                if len(chosen_route_copy) > chosen_n + 2:
                    swap_bin_5 = rsample(chosen_route_copy[1 : len(chosen_route_copy) - 4], 1)[0]
                    position_bin_5 = chosen_route.index(swap_bin_5)
                    chosen_route[position_bin_1], chosen_route[position_bin_5] = (
                        chosen_route[position_bin_5],
                        chosen_route[position_bin_1],
                    )
                    swap_bin_6 = chosen_route[position_bin_5 + 1]
                    position_bin_6 = chosen_route.index(swap_bin_6)
                    chosen_route[position_bin_2], chosen_route[position_bin_6] = (
                        chosen_route[position_bin_6],
                        chosen_route[position_bin_2],
                    )
                    swap_bin_7 = chosen_route[position_bin_5 + 2]
                    position_bin_7 = chosen_route.index(swap_bin_7)
                    chosen_route[position_bin_3], chosen_route[position_bin_7] = (
                        chosen_route[position_bin_7],
                        chosen_route[position_bin_3],
                    )
                    swap_bin_8 = chosen_route[position_bin_5 + 3]
                    position_bin_8 = chosen_route.index(swap_bin_8)
                    chosen_route[position_bin_4], chosen_route[position_bin_8] = (
                        chosen_route[position_bin_8],
                        chosen_route[position_bin_4],
                    )
            else:
                swap_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 5], 1)[0]
                position_bin_1 = chosen_route.index(swap_bin_1)
                swap_bin_2 = chosen_route[position_bin_1 + 1]
                position_bin_2 = chosen_route.index(swap_bin_2)
                swap_bin_3 = chosen_route[position_bin_1 + 2]
                position_bin_3 = chosen_route.index(swap_bin_3)
                swap_bin_4 = chosen_route[position_bin_1 + 3]
                position_bin_4 = chosen_route.index(swap_bin_4)
                swap_bin_5 = chosen_route[position_bin_1 + 4]
                position_bin_5 = chosen_route.index(swap_bin_5)
                chosen_route_copy = deepcopy(chosen_route)
                del chosen_route_copy[position_bin_1 - 4 : position_bin_1 + 5]
                if chosen_route_copy[0] != 0:
                    chosen_route_copy = [0] + chosen_route_copy

                if chosen_route_copy[-1] != 0:
                    chosen_route_copy = chosen_route_copy + [0]

                if len(chosen_route_copy) > chosen_n + 2:
                    swap_bin_6 = rsample(chosen_route_copy[1 : len(chosen_route_copy) - 5], 1)[0]
                    position_bin_6 = chosen_route.index(swap_bin_6)
                    chosen_route[position_bin_1], chosen_route[position_bin_6] = (
                        chosen_route[position_bin_6],
                        chosen_route[position_bin_1],
                    )
                    swap_bin_7 = chosen_route[position_bin_6 + 1]
                    position_bin_7 = chosen_route.index(swap_bin_7)
                    chosen_route[position_bin_2], chosen_route[position_bin_7] = (
                        chosen_route[position_bin_7],
                        chosen_route[position_bin_2],
                    )
                    swap_bin_8 = chosen_route[position_bin_6 + 2]
                    position_bin_8 = chosen_route.index(swap_bin_8)
                    chosen_route[position_bin_3], chosen_route[position_bin_8] = (
                        chosen_route[position_bin_8],
                        chosen_route[position_bin_3],
                    )
                    swap_bin_9 = chosen_route[position_bin_6 + 3]
                    position_bin_9 = chosen_route.index(swap_bin_9)
                    chosen_route[position_bin_4], chosen_route[position_bin_9] = (
                        chosen_route[position_bin_9],
                        chosen_route[position_bin_4],
                    )
                    swap_bin_10 = chosen_route[position_bin_6 + 4]
                    position_bin_10 = chosen_route.index(swap_bin_10)
                    chosen_route[position_bin_5], chosen_route[position_bin_10] = (
                        chosen_route[position_bin_10],
                        chosen_route[position_bin_5],
                    )
    return chosen_n


# Function to swap n random bins between two different routes
def swap_n_2_routes_random(routes_list):
    """
    Inter-route perturbation: Swap n random bins between two different routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0:
        if len(routes_list) > 1:
            chosen_route_1 = rsample(routes_list, 1)[0]
            chosen_route_2 = chosen_route_1
            while chosen_route_2 == chosen_route_1:
                chosen_route_2 = rsample(routes_list, 1)[0]

            if len(chosen_route_1) >= chosen_n + 2 and len(chosen_route_2) >= chosen_n + 2:
                if chosen_n == 2:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3 = bin_2
                    while bin_3 == bin_2:
                        bin_3 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_3_position = chosen_route_1.index(bin_3)
                    bin_4 = bin_1
                    while bin_4 == bin_1:
                        bin_4 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_4_position = chosen_route_2.index(bin_4)
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                elif chosen_n == 3:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3 = bin_2
                    while bin_3 == bin_2:
                        bin_3 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_3_position = chosen_route_1.index(bin_3)
                    bin_4 = bin_1
                    while bin_4 == bin_1:
                        bin_4 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_4_position = chosen_route_2.index(bin_4)
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    bin_5 = bin_2
                    while bin_5 == bin_2 or bin_5 == bin_4:
                        bin_5 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_5_position = chosen_route_1.index(bin_5)
                    bin_6 = bin_1
                    while bin_6 == bin_1 or bin_6 == bin_3:
                        bin_6 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_6_position = chosen_route_2.index(bin_6)
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                elif chosen_n == 4:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3 = bin_2
                    while bin_3 == bin_2:
                        bin_3 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_3_position = chosen_route_1.index(bin_3)
                    bin_4 = bin_1
                    while bin_4 == bin_1:
                        bin_4 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_4_position = chosen_route_2.index(bin_4)
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    bin_5 = bin_2
                    while bin_5 == bin_2 or bin_5 == bin_4:
                        bin_5 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_5_position = chosen_route_1.index(bin_5)
                    bin_6 = bin_1
                    while bin_6 == bin_1 or bin_6 == bin_3:
                        bin_6 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_6_position = chosen_route_2.index(bin_6)
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                    bin_7 = bin_2
                    while bin_7 == bin_2 or bin_7 == bin_4 or bin_7 == bin_6:
                        bin_7 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_7_position = chosen_route_1.index(bin_7)
                    bin_8 = bin_1
                    while bin_8 == bin_1 or bin_8 == bin_3 or bin_8 == bin_5:
                        bin_8 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_8_position = chosen_route_2.index(bin_8)
                    chosen_route_1[bin_7_position], chosen_route_2[bin_8_position] = (
                        chosen_route_2[bin_8_position],
                        chosen_route_1[bin_7_position],
                    )
                else:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3 = bin_2
                    while bin_3 == bin_2:
                        bin_3 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_3_position = chosen_route_1.index(bin_3)
                    bin_4 = bin_1
                    while bin_4 == bin_1:
                        bin_4 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_4_position = chosen_route_2.index(bin_4)
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    bin_5 = bin_2
                    while bin_5 == bin_2 or bin_5 == bin_4:
                        bin_5 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_5_position = chosen_route_1.index(bin_5)
                    bin_6 = bin_1
                    while bin_6 == bin_1 or bin_6 == bin_3:
                        bin_6 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_6_position = chosen_route_2.index(bin_6)
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                    bin_7 = bin_2
                    while bin_7 == bin_2 or bin_7 == bin_4 or bin_7 == bin_6:
                        bin_7 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_7_position = chosen_route_1.index(bin_7)
                    bin_8 = bin_1
                    while bin_8 == bin_1 or bin_8 == bin_3 or bin_8 == bin_5:
                        bin_8 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_8_position = chosen_route_2.index(bin_8)
                    chosen_route_1[bin_7_position], chosen_route_2[bin_8_position] = (
                        chosen_route_2[bin_8_position],
                        chosen_route_1[bin_7_position],
                    )
                    bin_9 = bin_2
                    while bin_9 == bin_2 or bin_9 == bin_4 or bin_9 == bin_6 or bin_9 == bin_8:
                        bin_9 = rsample(chosen_route_1[1 : len(chosen_route_1) - 1], 1)[0]

                    bin_9_position = chosen_route_1.index(bin_9)
                    bin_10 = bin_1
                    while bin_10 == bin_1 or bin_10 == bin_3 or bin_10 == bin_5 or bin_10 == bin_7:
                        bin_10 = rsample(chosen_route_2[1 : len(chosen_route_2) - 1], 1)[0]

                    bin_10_position = chosen_route_2.index(bin_10)
                    chosen_route_1[bin_9_position], chosen_route_2[bin_10_position] = (
                        chosen_route_2[bin_10_position],
                        chosen_route_1[bin_9_position],
                    )
    return chosen_n


# Function to swap n consecutive bins between two different routes
def swap_n_2_routes_consecutive(routes_list):
    """
    Inter-route perturbation: Swap two consecutive sequences of n bins
    between two different routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0:
        if len(routes_list) > 1:
            chosen_route_1 = rsample(routes_list, 1)[0]
            chosen_route_2 = chosen_route_1
            while chosen_route_2 == chosen_route_1:
                chosen_route_2 = rsample(routes_list, 1)[0]

            if len(chosen_route_1) >= chosen_n + 2 and len(chosen_route_2) >= chosen_n + 2:
                if chosen_n == 2:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 2], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 2], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    chosen_route_1[bin_1_position + 1]
                    bin_3_position = bin_1_position + 1
                    chosen_route_2[bin_2_position + 1]
                    bin_4_position = bin_2_position + 1
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                elif chosen_n == 3:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 3], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 3], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    chosen_route_1[bin_1_position + 1]
                    bin_3_position = bin_1_position + 1
                    chosen_route_2[bin_2_position + 1]
                    bin_4_position = bin_2_position + 1
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    chosen_route_1[bin_1_position + 2]
                    bin_5_position = bin_1_position + 2
                    chosen_route_2[bin_2_position + 2]
                    bin_6_position = bin_2_position + 2
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                elif chosen_n == 4:
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 4], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 4], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3_position = bin_1_position + 1
                    bin_4_position = bin_2_position + 1
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    bin_5_position = bin_1_position + 2
                    bin_6_position = bin_2_position + 2
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                    bin_7_position = bin_1_position + 3
                    bin_8_position = bin_2_position + 3
                    chosen_route_1[bin_7_position], chosen_route_2[bin_8_position] = (
                        chosen_route_2[bin_8_position],
                        chosen_route_1[bin_7_position],
                    )
                else:  # chosen_n == 5
                    bin_1 = rsample(chosen_route_1[1 : len(chosen_route_1) - 5], 1)[0]
                    bin_1_position = chosen_route_1.index(bin_1)
                    bin_2 = rsample(chosen_route_2[1 : len(chosen_route_2) - 5], 1)[0]
                    bin_2_position = chosen_route_2.index(bin_2)
                    chosen_route_1[bin_1_position], chosen_route_2[bin_2_position] = (
                        chosen_route_2[bin_2_position],
                        chosen_route_1[bin_1_position],
                    )
                    bin_3_position = bin_1_position + 1
                    bin_4_position = bin_2_position + 1
                    chosen_route_1[bin_3_position], chosen_route_2[bin_4_position] = (
                        chosen_route_2[bin_4_position],
                        chosen_route_1[bin_3_position],
                    )
                    bin_5_position = bin_1_position + 2
                    bin_6_position = bin_2_position + 2
                    chosen_route_1[bin_5_position], chosen_route_2[bin_6_position] = (
                        chosen_route_2[bin_6_position],
                        chosen_route_1[bin_5_position],
                    )
                    bin_7_position = bin_1_position + 3
                    bin_8_position = bin_2_position + 3
                    chosen_route_1[bin_7_position], chosen_route_2[bin_8_position] = (
                        chosen_route_2[bin_8_position],
                        chosen_route_1[bin_7_position],
                    )
                    bin_9_position = bin_1_position + 4
                    bin_10_position = bin_2_position + 4
                    chosen_route_1[bin_9_position], chosen_route_2[bin_10_position] = (
                        chosen_route_2[bin_10_position],
                        chosen_route_1[bin_9_position],
                    )
    return chosen_n
