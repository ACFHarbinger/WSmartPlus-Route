"""
Node relocation operators for intra-route and inter-route perturbations.

Contains primary mutation procedures that move single or multiple bins
randomly or consecutively within the same route or between different routes.
These operators form the core of the local search neighborhood exploration.
"""

from random import sample as rsample

__all__ = [
    "move_1_route",
    "move_2_routes",
    "move_n_route_random",
    "move_n_route_consecutive",
    "move_n_2_routes_random",
    "move_n_2_routes_consecutive",
]


# Function to move 1 route (Choose one route and move one bin from one place in the route to another place in the route)
def move_1_route(routes_list):
    """
    Intra-route perturbation: Move one bin to a different position in the same route.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        if len(chosen_route) > 3:
            bin_to_move = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
            position_bin = bin_to_move
            while position_bin == bin_to_move:
                position_bin = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

            position = chosen_route.index(position_bin)
            chosen_route.remove(bin_to_move)
            chosen_route.insert(position, bin_to_move)


# Function to move one bin from one route to another route
# (Choose two routes and move one element from one route to the other route)
def move_2_routes(routes_list):
    """
    Inter-route perturbation: Move one bin from one route to another.

    Args:
        routes_list (List[List[int]]): Current routing solution.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0:
        if len(routes_list) > 1:
            chosen_route_1 = rsample(routes_list, 1)[0]
            chosen_route_2 = chosen_route_1
            while chosen_route_2 == chosen_route_1:
                chosen_route_2 = rsample(routes_list, 1)[0]

            two_routes = []
            two_routes.append(chosen_route_1)
            two_routes.append(chosen_route_2)

            donor_route = rsample(two_routes, 1)[0]
            donor_route_index = set([two_routes.index(donor_route)])

            indexes = set([0, 1])
            receptor_route_index = list(indexes - donor_route_index)[0]
            receptor_route = two_routes[receptor_route_index]

            donor_bin = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]
            receptor_position = receptor_route.index(rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0])

            donor_route.remove(donor_bin)
            receptor_route.insert(receptor_position, donor_bin)
            if len(donor_route) == 2:
                routes_list.remove(donor_route)

        routes_list = list(filter(None, routes_list))


# Function to move n random bins inside a route
def move_n_route_random(routes_list):
    """
    Intra-route perturbation: Move n random bins to new positions within their routes.

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
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                position_bin_2 = bin_to_move_2
                while position_bin_2 == bin_to_move_2:
                    position_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_2 = chosen_route.index(position_bin_2)
                chosen_route.remove(bin_to_move_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.insert(position_2, bin_to_move_2)
            elif chosen_n == 3:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                position_bin_2 = bin_to_move_2
                while position_bin_2 == bin_to_move_2:
                    position_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_2 = chosen_route.index(position_bin_2)
                position_bin_3 = bin_to_move_3
                while position_bin_3 == bin_to_move_3:
                    position_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_3 = chosen_route.index(position_bin_3)
                chosen_route.remove(bin_to_move_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.insert(position_2, bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                chosen_route.insert(position_3, bin_to_move_3)
            elif chosen_n == 4:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                position_bin_2 = bin_to_move_2
                while position_bin_2 == bin_to_move_2:
                    position_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_2 = chosen_route.index(position_bin_2)
                position_bin_3 = bin_to_move_3
                while position_bin_3 == bin_to_move_3:
                    position_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_3 = chosen_route.index(position_bin_3)
                position_bin_4 = bin_to_move_4
                while position_bin_4 == bin_to_move_4:
                    position_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_4 = chosen_route.index(position_bin_4)
                chosen_route.remove(bin_to_move_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.insert(position_2, bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                chosen_route.insert(position_3, bin_to_move_3)
                chosen_route.remove(bin_to_move_4)
                chosen_route.insert(position_4, bin_to_move_4)
            else:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                bin_to_move_5 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                position_bin_2 = bin_to_move_2
                while position_bin_2 == bin_to_move_2:
                    position_bin_2 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_2 = chosen_route.index(position_bin_2)
                position_bin_3 = bin_to_move_3
                while position_bin_3 == bin_to_move_3:
                    position_bin_3 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_3 = chosen_route.index(position_bin_3)
                position_bin_4 = bin_to_move_4
                while position_bin_4 == bin_to_move_4:
                    position_bin_4 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                position_4 = chosen_route.index(position_bin_4)

                position_bin_5 = bin_to_move_5
                while position_bin_5 == bin_to_move_5:
                    position_bin_5 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_5 = chosen_route.index(position_bin_5)
                chosen_route.remove(bin_to_move_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.insert(position_2, bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                chosen_route.insert(position_3, bin_to_move_3)
                chosen_route.remove(bin_to_move_4)
                chosen_route.insert(position_4, bin_to_move_4)
                chosen_route.remove(bin_to_move_5)
                chosen_route.insert(position_5, bin_to_move_5)
    return chosen_n


# Function to move n consecutive bins inside a route
def move_n_route_consecutive(routes_list):
    """
    Intra-route perturbation: Move a sequence of consecutive bins to a new position
    in the same route.

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
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 2], 1)[0]
                bin_1_position = chosen_route.index(bin_to_move_1)
                bin_to_move_2 = chosen_route[bin_1_position + 1]
                chosen_route.remove(bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.insert(position_1 + 1, bin_to_move_2)
            elif chosen_n == 3:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 3], 1)[0]
                bin_1_position = chosen_route.index(bin_to_move_1)
                bin_to_move_2 = chosen_route[bin_1_position + 1]
                bin_to_move_3 = chosen_route[bin_1_position + 2]
                chosen_route.remove(bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.insert(position_1 + 1, bin_to_move_2)
                chosen_route.insert(position_1 + 2, bin_to_move_3)
            elif chosen_n == 4:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 4], 1)[0]
                bin_1_position = chosen_route.index(bin_to_move_1)
                bin_to_move_2 = chosen_route[bin_1_position + 1]
                bin_to_move_3 = chosen_route[bin_1_position + 2]
                bin_to_move_4 = chosen_route[bin_1_position + 3]
                chosen_route.remove(bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                chosen_route.remove(bin_to_move_4)
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.insert(position_1 + 1, bin_to_move_2)
                chosen_route.insert(position_1 + 2, bin_to_move_3)
                chosen_route.insert(position_1 + 3, bin_to_move_4)
            else:
                bin_to_move_1 = rsample(chosen_route[1 : len(chosen_route) - 5], 1)[0]
                bin_1_position = chosen_route.index(bin_to_move_1)
                bin_to_move_2 = chosen_route[bin_1_position + 1]
                bin_to_move_3 = chosen_route[bin_1_position + 2]
                bin_to_move_4 = chosen_route[bin_1_position + 3]
                bin_to_move_5 = chosen_route[bin_1_position + 4]
                chosen_route.remove(bin_to_move_1)
                chosen_route.remove(bin_to_move_2)
                chosen_route.remove(bin_to_move_3)
                chosen_route.remove(bin_to_move_4)
                chosen_route.remove(bin_to_move_5)
                position_bin_1 = bin_to_move_1
                while position_bin_1 == bin_to_move_1:
                    position_bin_1 = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                position_1 = chosen_route.index(position_bin_1)
                chosen_route.insert(position_1, bin_to_move_1)
                chosen_route.insert(position_1 + 1, bin_to_move_2)
                chosen_route.insert(position_1 + 2, bin_to_move_3)
                chosen_route.insert(position_1 + 3, bin_to_move_4)
                chosen_route.insert(position_1 + 4, bin_to_move_5)
    return chosen_n


# Function to move several bins from one route to another route randomly
def move_n_2_routes_random(routes_list):
    """
    Inter-route perturbation: Move n random bins from one route to another.

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

            two_routes = []
            two_routes.append(chosen_route_1)
            two_routes.append(chosen_route_2)

            donor_route = rsample(two_routes, 1)[0]
            donor_route_index = set([two_routes.index(donor_route)])

            indexes = set([0, 1])
            receptor_route_index = list(indexes - donor_route_index)[0]
            receptor_route = two_routes[receptor_route_index]
            if len(donor_route) >= chosen_n + 2:
                if chosen_n == 2:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_2 = donor_bin_1
                    while donor_bin_2 == donor_bin_1:
                        donor_bin_2 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)

                    receptor_position_2 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_2, donor_bin_2)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                elif chosen_n == 3:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]
                    donor_bin_2 = donor_bin_1
                    while donor_bin_2 == donor_bin_1:
                        donor_bin_2 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_3 = donor_bin_1
                    while donor_bin_3 == donor_bin_1 or donor_bin_3 == donor_bin_2:
                        donor_bin_3 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)

                    receptor_position_2 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_2, donor_bin_2)

                    receptor_position_3 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_3, donor_bin_3)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                elif chosen_n == 4:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]
                    donor_bin_2 = donor_bin_1
                    while donor_bin_2 == donor_bin_1:
                        donor_bin_2 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_3 = donor_bin_1
                    while donor_bin_3 == donor_bin_1 or donor_bin_3 == donor_bin_2:
                        donor_bin_3 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_4 = donor_bin_1
                    while donor_bin_4 == donor_bin_1 or donor_bin_4 == donor_bin_2 or donor_bin_4 == donor_bin_3:
                        donor_bin_4 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)

                    receptor_position_2 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_2, donor_bin_2)

                    receptor_position_3 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_3, donor_bin_3)

                    receptor_position_4 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_4)
                    receptor_route.insert(receptor_position_4, donor_bin_4)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                else:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]
                    donor_bin_2 = donor_bin_1
                    while donor_bin_2 == donor_bin_1:
                        donor_bin_2 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_3 = donor_bin_1
                    while donor_bin_3 == donor_bin_1 or donor_bin_3 == donor_bin_2:
                        donor_bin_3 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_4 = donor_bin_1
                    while donor_bin_4 == donor_bin_1 or donor_bin_4 == donor_bin_2 or donor_bin_4 == donor_bin_3:
                        donor_bin_4 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    donor_bin_5 = donor_bin_1
                    while (
                        donor_bin_5 == donor_bin_1
                        or donor_bin_5 == donor_bin_2
                        or donor_bin_5 == donor_bin_3
                        or donor_bin_5 == donor_bin_4
                    ):
                        donor_bin_5 = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)

                    receptor_position_2 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_2, donor_bin_2)

                    receptor_position_3 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_3, donor_bin_3)

                    receptor_position_4 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_4)
                    receptor_route.insert(receptor_position_4, donor_bin_4)

                    receptor_position_5 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )
                    donor_route.remove(donor_bin_5)
                    receptor_route.insert(receptor_position_5, donor_bin_5)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
        routes_list = list(filter(None, routes_list))
    return chosen_n


# function to move several bins from one route to another route consecutively
def move_n_2_routes_consecutive(routes_list):
    """
    Inter-route perturbation: Move a sequence of consecutive bins from one route
    to another.

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

            two_routes = []
            two_routes.append(chosen_route_1)
            two_routes.append(chosen_route_2)

            donor_route = rsample(two_routes, 1)[0]
            donor_route_index = set([two_routes.index(donor_route)])

            indexes = set([0, 1])
            receptor_route_index = list(indexes - donor_route_index)[0]
            receptor_route = two_routes[receptor_route_index]
            if len(donor_route) >= chosen_n + 2:
                if chosen_n == 2:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 2], 1)[0]
                    donor_bin_1_position = donor_route.index(donor_bin_1)
                    donor_bin_2 = donor_route[donor_bin_1_position + 1]
                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )

                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_1 + 1, donor_bin_2)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                elif chosen_n == 3:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 3], 1)[0]
                    donor_bin_1_position = donor_route.index(donor_bin_1)
                    donor_bin_2 = donor_route[donor_bin_1_position + 1]
                    donor_bin_3 = donor_route[donor_bin_1_position + 2]
                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )

                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_1 + 1, donor_bin_2)
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_1 + 2, donor_bin_3)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                elif chosen_n == 4:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 4], 1)[0]
                    donor_bin_1_position = donor_route.index(donor_bin_1)
                    donor_bin_2 = donor_route[donor_bin_1_position + 1]
                    donor_bin_3 = donor_route[donor_bin_1_position + 2]
                    donor_bin_4 = donor_route[donor_bin_1_position + 3]
                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )

                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_1 + 1, donor_bin_2)
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_1 + 2, donor_bin_3)
                    donor_route.remove(donor_bin_4)
                    receptor_route.insert(receptor_position_1 + 3, donor_bin_4)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
                else:
                    donor_bin_1 = rsample(donor_route[1 : len(donor_route) - 5], 1)[0]
                    donor_bin_1_position = donor_route.index(donor_bin_1)
                    donor_bin_2 = donor_route[donor_bin_1_position + 1]
                    donor_bin_3 = donor_route[donor_bin_1_position + 2]
                    donor_bin_4 = donor_route[donor_bin_1_position + 3]
                    donor_bin_5 = donor_route[donor_bin_1_position + 4]
                    receptor_position_1 = receptor_route.index(
                        rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
                    )

                    donor_route.remove(donor_bin_1)
                    receptor_route.insert(receptor_position_1, donor_bin_1)
                    donor_route.remove(donor_bin_2)
                    receptor_route.insert(receptor_position_1 + 1, donor_bin_2)
                    donor_route.remove(donor_bin_3)
                    receptor_route.insert(receptor_position_1 + 2, donor_bin_3)
                    donor_route.remove(donor_bin_4)
                    receptor_route.insert(receptor_position_1 + 3, donor_bin_4)
                    donor_route.remove(donor_bin_5)
                    receptor_route.insert(receptor_position_1 + 4, donor_bin_5)
                    if len(donor_route) == 2:
                        routes_list.remove(donor_route)
        routes_list = list(filter(None, routes_list))
    return chosen_n
