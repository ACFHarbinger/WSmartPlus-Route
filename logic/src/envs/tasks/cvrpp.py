"""cvrpp.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import cvrpp
"""

import torch

from logic.src.envs.tasks.vrpp import VRPP


class CVRPP(VRPP):
    """
    Capacitated VRPP.
    Includes vehicle capacity constraints.
    """

    NAME = "cvrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute CVRPP costs (same as VRPP but checks capacity).

        Args:
            dataset: Problem data.
            pi: Tours.
            cw_dict: Cost weights.
            dist_matrix: optional.

        Returns:
            Tuple of (cost, dict, None).
        """
        cost, c_dict, _ = VRPP.get_costs(dataset, pi, cw_dict, dist_matrix)

        # CVRPP specific: Check total capacity PER TRIP
        capacity = dataset.get("capacity", dataset.get("max_waste", torch.tensor(100.0)))

        # Extract trip waste
        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        # For each sequence in pi, calculate cumulative waste and reset at 0
        w = waste_with_depot.gather(1, pi)

        # Simple loop-based trip check for robustness in tests
        for b in range(pi.size(0)):
            cur_trip_waste = 0
            for i in range(pi.size(1)):
                node = pi[b, i].item()
                if node == 0:
                    cur_trip_waste = 0
                else:
                    cur_trip_waste += w[b, i].item()
                    # Use a small epsilon for float comparison
                    assert cur_trip_waste <= capacity[b].item() + 1e-6, "Used more than capacity"

        return cost, c_dict, _
