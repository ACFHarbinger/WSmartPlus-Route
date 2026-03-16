"""
Simulator-related constants for the GUI.
"""

# Simulator Data
COUNTY_AREAS = {
    "Rio Maior": "riomaior",
    "Figueira da Foz": "figdafoz",
    "Mixtura de Freguesias": "mixrmbac",
}

WASTE_TYPES = ["", "Paper", "Plastic", "Glass"]

# Test Simulator
SIMULATOR_TEST_POLICIES = {
    "Attention Model": "am",
    "Attention Model with Graph Connections": "amgc",
    "TransformerGCN Model": "transgcn",
    "Deep Decoder Attention Model": "ddam",
    "Gurobi SWC-TCF Solver": "gurobi_swc_tcf",
    "Hexaly SWC-TCF Solver": "hexaly_swc_tcf",
    "Look-Ahead Policy": "policy_look_ahead",
    "Look-Ahead SWC-TCF Policy": "policy_look_ahead_swc_tcf",
    "Look-Ahead SANS Policy": "policy_look_ahead_sans",
    "Last Minute and Path Policy": "policy_last_minute_and_path",
    "Last Minute Policy": "policy_last_minute",
    "Regular Policy": "policy_regular",
}

DISTANCE_MATRIX_METHODS = {
    "Google Maps (GMaps)": "gmaps",
    "Open Street Maps (OSM)": "osm",
    "Geo-Pandas Distance (GPD)": "gpd",
    "Geodesic Distance (GdsC)": "gdsc",
    "Haversine Distance (HsD)": "hsd",
    "Original Distance (OgD)": "ogd",
}
