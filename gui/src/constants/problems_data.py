"""
Problem types and data distribution constants for the GUI.
"""

# Problems and Data
PROBLEM_TYPES = ["VRPP", "CVRPP", "WCVRP", "CWCVRP", "SDWCVRP", "SCWCVRP"]

DATA_DIST_PROBLEMS = ["VRPP", "WCVRP"]

DATA_DISTRIBUTIONS = {
    "Empty": "empty",
    "Gamma 1": "gamma1",
    "Gamma 2": "gamma2",
    "Gamma 3": "gamma3",
    "Empirical": "emp",
    "Constant": "const",
    "Uniform": "unif",
    "Distance": "dist",
}
