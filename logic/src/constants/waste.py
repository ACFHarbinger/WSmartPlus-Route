"""
Waste management specific constants.
"""
from typing import Dict

# Waste management information
MAP_DEPOTS: Dict[str, str] = {
    "mixrmbac": "CTEASO",  # Rio Maior, Bombarral, Azambuja, Cadaval
    "riomaior": "CTEASO",
    "figueiradafoz": "CITVRSU",
}

WASTE_TYPES: Dict[str, str] = {
    "glass": "Embalagens de Vidro",
    "plastic": "Mistura de embalagens",
    "paper": "Embalagens de papel e cartão",
}
