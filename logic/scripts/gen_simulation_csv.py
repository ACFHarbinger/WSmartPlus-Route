"""
Parse simulation output directories into a summary CSV.

Walks an output directory tree of the form:

    <root>/
      <area><N>_plastic/          e.g. riomaior100_plastic
        <dist>/                   e.g. emp, gamma3
          <run_name>/             e.g. la_cls, lm_ftsp
            log_<policy>_1N.json

and writes a CSV with the same column schema used by
public/global/simulation/simulation_summary.csv.

Usage
-----
    uv run python logic/scripts/gen_simulation_csv.py \\
        --output-dir assets/output/90days \\
        --out-csv public/global/simulation/simulation_summary_90d.csv

    # 30-day (re-generate from raw output):
    uv run python logic/scripts/gen_simulation_csv.py \\
        --output-dir assets/output/30days \\
        --out-csv public/global/simulation/simulation_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ── Directory → city / N ───────────────────────────────────────────────────────
_DIR_CITY: dict[str, tuple[str, int]] = {
    "riomaior100":      ("Rio Maior", 100),
    "riomaior170":      ("Rio Maior", 170),
    "figueiradafoz350": ("Figueira da Foz", 350),
}

_DIST_MAP: dict[str, str] = {
    "emp":    "Empirical",
    "gamma1": "Gamma-1",
    "gamma2": "Gamma-2",
    "gamma3": "Gamma-3",
}

# ── Filename parsing ────────────────────────────────────────────────────────────
# Strategy prefixes and their (strategy, cf, sl_var) values
_STRATEGY_PREFIXES: list[tuple[str, str, str | None, str | None]] = [
    ("last_minute_cf70_", "LM",  "CF70", None),
    ("last_minute_cf90_", "LM",  "CF90", None),
    ("service_level1_",   "SL",  None,   "SL1"),
    ("service_level2_",   "SL",  None,   "SL2"),
    ("lookahead_",        "LA",  None,   None),
]

# Known constructor identifiers (longest-match first to avoid prefix collisions)
_CONSTRUCTORS: list[tuple[str, str]] = [
    ("pg_clns", "PG-CLNS"),
    ("swc_tcf", "SWC-TCF"),
    ("aco_hh",  "ACO_HH"),
    ("psoma",   "PSOMA"),
    ("sans",    "SANS"),
    ("alns",    "ALNS"),
    ("bpc",     "BPC"),
    ("hgs",     "HGS"),
]

_IMPROVER_PAT = re.compile(r"_(cls|ftsp)_\d+N$", re.IGNORECASE)


def _parse_filename(stem: str) -> dict | None:
    """
    Parse log filename stem (no extension) into metadata dict.

    Expected form: log_{strategy_tokens}_{constructor_tokens}_{improver}_{N}N
    """
    if not stem.startswith("log_"):
        return None
    rest = stem[4:]  # strip "log_"

    # Detect strategy
    strategy = cf = sl_var = None
    for prefix, strat, cf_val, sl_val in _STRATEGY_PREFIXES:
        if rest.startswith(prefix):
            strategy, cf, sl_var = strat, cf_val, sl_val
            rest = rest[len(prefix):]
            break
    if strategy is None:
        return None

    # Detect improver suffix
    m = _IMPROVER_PAT.search(rest)
    if not m:
        return None
    improver = m.group(1).upper()
    middle = rest[: m.start()]  # e.g. "bpc_custom" or "aco_hh_custom_ftsp"

    # Detect constructor from known prefixes
    constructor = None
    for token, label in _CONSTRUCTORS:
        if middle.startswith(token):
            constructor = label
            break
    if constructor is None:
        return None

    return {
        "strategy": strategy,
        "cf":       cf,
        "sl_var":   sl_var,
        "improver": improver,
        "constructor": constructor,
    }


def _parse_area_dir(dirname: str) -> tuple[str, int] | None:
    """Map e.g. 'riomaior100_plastic' → ('Rio Maior', 100)."""
    key = dirname.replace("_plastic", "")
    return _DIR_CITY.get(key)


def parse_output_dir(output_dir: Path) -> pd.DataFrame:
    """Walk *output_dir* and return a DataFrame of simulation metrics."""
    rows: list[dict] = []

    for area_dir in sorted(output_dir.iterdir()):
        if not area_dir.is_dir():
            continue
        city_n = _parse_area_dir(area_dir.name)
        if city_n is None:
            continue
        city, N = city_n

        for dist_dir in sorted(area_dir.iterdir()):
            if not dist_dir.is_dir():
                continue
            dist = _DIST_MAP.get(dist_dir.name)
            if dist is None:
                continue

            for run_dir in sorted(dist_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                for log_file in sorted(run_dir.glob("log_*.json")):
                    meta = _parse_filename(log_file.stem)
                    if meta is None:
                        continue

                    try:
                        data = json.loads(log_file.read_text())
                    except Exception:
                        continue

                    mean = data.get("mean", {})
                    if not mean:
                        continue

                    rows.append({
                        "city":        city,
                        "N":           N,
                        "dist":        dist,
                        "improver":    meta["improver"],
                        "strategy":    meta["strategy"],
                        "cf":          meta["cf"] or "",
                        "sl_var":      meta["sl_var"] or "",
                        "constructor": meta["constructor"],
                        "overflows":   mean.get("overflows",  0),
                        "kg":          mean.get("kg",          0),
                        "ncol":        mean.get("ncol",        0),
                        "kg_lost":     mean.get("kg_lost",     0),
                        "km":          mean.get("km",          0),
                        "kgkm":        mean.get("kg/km",       0),
                        "reward":      mean.get("reward",      0),
                        "profit":      mean.get("profit",      0),
                        "time":        mean.get("time",        0),
                        "days":        mean.get("days",        0),
                    })

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="assets/output/90days",
                   help="Root of the simulation output tree")
    p.add_argument("--out-csv", default="public/global/simulation/simulation_summary_90d.csv",
                   help="Destination CSV path")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    out_csv = Path(args.out_csv)

    if not output_dir.is_dir():
        raise SystemExit(f"Output dir not found: {output_dir}")

    print(f"Parsing: {output_dir}")
    df = parse_output_dir(output_dir)

    if df.empty:
        raise SystemExit("No log files found — check the output directory structure.")

    print(f"  Rows: {len(df)}")
    print(f"  Cities: {sorted(df['city'].unique())}")
    print(f"  Distributions: {sorted(df['dist'].unique())}")
    print(f"  Strategies: {sorted(df['strategy'].unique())}")
    print(f"  Improvers: {sorted(df['improver'].unique())}")
    print(f"  Constructors: {sorted(df['constructor'].unique())}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")


if __name__ == "__main__":
    main()
