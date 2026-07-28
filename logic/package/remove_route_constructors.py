"""Script to clean up Route Constructor components."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .cleanup_helper import ROUTE_CONSTRUCTORS, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_route_constructors.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        # Support comma separated input as well
        for acr in acronym.split(','):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up Route Constructor: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=ROUTE_CONSTRUCTORS["yaml_dirs"],
                config_dirs=ROUTE_CONSTRUCTORS["config_dirs"],
                impl_dirs=ROUTE_CONSTRUCTORS["impl_dirs"],
                yaml_prefixes=ROUTE_CONSTRUCTORS["yaml_prefixes"],
                config_prefixes=ROUTE_CONSTRUCTORS["config_prefixes"]
            )

if __name__ == "__main__":
    main()
