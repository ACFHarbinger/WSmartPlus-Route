"""Script to clean up RL Training Algorithm components."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleanup_helper import RL_ALGORITHMS, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_rl_algorithms.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        for acr in acronym.split(','):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up RL Algorithm: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=RL_ALGORITHMS["yaml_dirs"],
                config_dirs=RL_ALGORITHMS["config_dirs"],
                impl_dirs=RL_ALGORITHMS["impl_dirs"],
                yaml_prefixes=RL_ALGORITHMS["yaml_prefixes"],
                config_prefixes=RL_ALGORITHMS["config_prefixes"]
            )

if __name__ == "__main__":
    main()
