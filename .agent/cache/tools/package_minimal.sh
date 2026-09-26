#!/usr/bin/env bash
# Build the file listing and zip archive for the minimal export branch.
# Only git-tracked files are packaged (plus the GridBase subset of the submodule and a
# small runtime data subset), so local build artefacts never leak in.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
STAMP="$(date +%Y%m%d)"
OUT_DIR="$ROOT/dist"
STAGE="$(mktemp -d)/wsmart-route-minimal"
mkdir -p "$OUT_DIR" "$STAGE"

# 1. tracked files (branch content) + the GridBase subset of the submodule
# .agent/ (agent bus, review report, repro tools) is internal and stays out of the handoff.
git ls-files -z -- . ':(exclude).agent' | xargs -0 -I{} bash -c '[ -d "$2" ] && exit 0; mkdir -p "$1/$(dirname "$2")" && cp -p "$2" "$1/$2"' _ "$STAGE" {}

# wsmart_bin_analysis (AGPL-3.0 git submodule): the retained code imports only
# GridBase, so ship its runtime closure instead of the whole repo (owner D11).
SUB=logic/src/pipeline/simulations/wsmart_bin_analysis
SUB_FILES=(
  LICENSE
  export/__init__.py export/grid.py export/save_load.py export/container.py
  export/enums/__init__.py export/enums/tag.py
  export/modules/core.py export/modules/analysis_utils.py
  export/modules/processing_utils.py export/modules/plotting_utils.py
)
SUB_REV="$(git -C "$SUB" rev-parse HEAD 2>/dev/null || true)"
if [ -z "$SUB_REV" ] || [ ! -f "$SUB/export/grid.py" ]; then
  echo "[package] $SUB is not checked out (git submodule update --init)" >&2; exit 1
fi
for f in "${SUB_FILES[@]}"; do
  mkdir -p "$STAGE/$SUB/$(dirname "$f")"; cp -p "$SUB/$f" "$STAGE/$SUB/$f"
done
cat > "$STAGE/$SUB/__init__.py" <<PYEOF
"""
Vendored subset of wsmart_bin_analysis (https://github.com/ACFHarbinger/wsmart_bin_analysis,
commit $SUB_REV, AGPL-3.0, see LICENSE in this directory).

Only GridBase and its import closure are shipped; the empirical waste
distribution samples daily fill rates through it.

Attributes:
    GridBase: Class for representing a grid of bins.
"""

from .export.grid import GridBase as GridBase

__all__ = ["GridBase"]
PYEOF
# plotting is interactive-only; matplotlib lives in the optional viz extra
python3 - "$STAGE/$SUB/export/container.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "from .modules.plotting_utils import VisualizationMixin\n"
new = (
    "try:\n"
    "    from .modules.plotting_utils import VisualizationMixin\n"
    "except ImportError:  # matplotlib is optional (viz extra); plotting is unavailable\n\n"
    "    class VisualizationMixin:  # type: ignore[no-redef]\n"
    "        pass\n"
)
assert s.count(old) == 1, "container.py import line changed upstream"
open(p, "w").write(s.replace(old, new))
PYEOF

# 2. minimal runtime data subset for Rio Maior (20/50/100/170-bin graphs)
DATA_FILES=(
  "data/simulator/distance_matrix/gmaps_distmat_plastic[riomaior].csv"
  "data/simulator/distance_matrix/submatrix/gmaps_distmat20_plastic[riomaior].csv"
  "data/simulator/distance_matrix/submatrix/gmaps_distmat50_plastic[riomaior].csv"
  "data/simulator/distance_matrix/submatrix/gmaps_distmat100_plastic[riomaior].csv"
  "data/simulator/distance_matrix/submatrix/gmaps_distmat170_plastic[riomaior].csv"
  "data/simulator/bins_selection/graphs_20V_1N_plastic.json"
  "data/simulator/bins_selection/graphs_50V_1N_plastic.json"
  "data/simulator/bins_selection/graphs_100V_1N_plastic.json"
  "data/simulator/bins_selection/graphs_170V_1N_plastic.json"
  "data/simulator/coordinates/Facilities.csv"
  "data/simulator/coordinates/old_out_info[riomaior].csv"
  "data/simulator/coordinates/out_info[riomaior].csv"
  "data/simulator/bins_waste/old_out_crude_rate[riomaior].csv"
  "data/simulator/bins_waste/out_rate_crude[riomaior].csv"
  "data/simulator/daily_waste/riomaior20_emp_wsr31_N10_seed42.pkl"
  "data/simulator/daily_waste/riomaior50_emp_wsr31_N10_seed42.pkl"
)
for f in "${DATA_FILES[@]}"; do
  if [ -f "$f" ]; then mkdir -p "$STAGE/$(dirname "$f")"; cp -p "$f" "$STAGE/$f"; else echo "[package] missing data file (skipped): $f"; fi
done

# 3. file listing (tree) inside the archive and next to it
( cd "$STAGE" && tree --charset ascii --noreport -a -I '__pycache__' . ) > "$STAGE/FILE_LIST.txt"
cp "$STAGE/FILE_LIST.txt" "$OUT_DIR/FILE_LIST.txt"

# 4. zip
ZIP="$OUT_DIR/wsmart-route-minimal-$STAMP.zip"
rm -f "$ZIP"
( cd "$(dirname "$STAGE")" && zip -q -r -X "$ZIP" "$(basename "$STAGE")" -x '*/__pycache__/*' )
echo "[package] archive: $ZIP ($(du -h "$ZIP" | cut -f1), $(unzip -l "$ZIP" | tail -1 | awk '{print $2}') files)"
echo "[package] listing: $OUT_DIR/FILE_LIST.txt"
rm -rf "$(dirname "$STAGE")"
