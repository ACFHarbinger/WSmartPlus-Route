"""Script to clean up Environment (Envs) components."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .cleanup_helper import ENVS, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_envs.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        for acr in acronym.split(","):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up Environment: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=ENVS["yaml_dirs"],
                config_dirs=ENVS["config_dirs"],
                impl_dirs=ENVS["impl_dirs"],
                yaml_prefixes=ENVS["yaml_prefixes"],
                config_prefixes=ENVS["config_prefixes"],
            )


if __name__ == "__main__":
    main()
