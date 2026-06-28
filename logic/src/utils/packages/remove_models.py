"""Script to clean up Model components."""

import os
import sys

# Ensure cleanup_helper is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .cleanup_helper import MODELS, clean_by_acronym


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_models.py <acronyms...>")
        sys.exit(1)

    acronyms = sys.argv[1:]
    for acronym in acronyms:
        for acr in acronym.split(','):
            acr = acr.strip()
            if not acr:
                continue
            print(f"\n--- Cleaning up Model: {acr} ---")
            clean_by_acronym(
                acronym=acr,
                yaml_dirs=MODELS["yaml_dirs"],
                config_dirs=MODELS["config_dirs"],
                impl_dirs=MODELS["impl_dirs"],
                yaml_prefixes=MODELS["yaml_prefixes"],
                config_prefixes=MODELS["config_prefixes"]
            )

if __name__ == "__main__":
    main()
