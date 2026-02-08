"""parameters.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import parameters
    """
from typing import Tuple


def load_area_and_waste_type_params(area: str, waste_type: str) -> Tuple[float, float, float, float, float]:
    """
    Retrieves area and waste-type specific simulation parameters.

    Returns physical and economic parameters calibrated for specific
    geographic areas and waste streams. Values are based on real-world
    data from Portuguese waste management operations.

    Args:
        area: Geographic area ('Rio Maior', 'Figueira da Foz', etc.)
        waste_type: Waste stream ('paper', 'plastic', 'glass')

    Returns:
        Tuple containing:
            - vehicle_capacity: Max bin capacity units per vehicle (%)
            - revenue: Revenue per kg of collected waste (€/kg)
            - density: Waste density (kg/L)
            - expenses: Cost per km traveled (€/km)
            - bin_volume: Individual bin volume (L)

    Raises:
        AssertionError: If waste_type or area not recognized
    """
    expenses = 1.0
    bin_volume = 2.5
    src_area = area.translate(str.maketrans("", "", "-_ ")).lower()

    if waste_type == "paper":
        revenue = 0.65 * 250 / 1000
        if src_area in ["riomaior", "mixrmbac"]:
            density = 21.0
            vehicle_capacity = 4000.0
        else:
            assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
            density = 32.0
            vehicle_capacity = 3000.0

    elif waste_type == "plastic":
        revenue = 0.65 * 898 / 1000
        if src_area in ["riomaior", "mixrmbac"]:
            density = 19.0
            vehicle_capacity = 3500.0
        else:
            assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
            density = 20.0
            vehicle_capacity = 2500.0

    else:
        assert waste_type == "glass", f"Unknown waste type: {waste_type}"
        revenue = 0.90 * 84 / 1000
        if src_area in ["riomaior", "mixrmbac"]:
            density = 190.0
            vehicle_capacity = 9000.0
        else:
            assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
            density = 200.0
            vehicle_capacity = 8000.0

    # Calculate percentage capacity
    vehicle_capacity = (vehicle_capacity / (bin_volume * density)) * 100
    return (vehicle_capacity, revenue, density, expenses, bin_volume)
