"""Lane-G assist probe (Muse, 2026-09-25): re-verifiable static checks behind §5.M.

Run from the workspace root on the minimal-export review branch::

    .venv/bin/python .agent/cache/tools/muse_lane_g_probe_20260925.py

Everything here is read-only (grep-equivalents + ``git show main:...``).
Exit 0 = all Muse §5.M claims reproduce as written. Exit != 0 prints the
first mismatch. Analysed commit for tree paths: ``70e660b03``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGIC = ROOT / "logic"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def py_files(exclude_submodule: bool = True):
    for p in list(LOGIC.rglob("*.py")) + [ROOT / "main.py"]:
        if exclude_submodule and "wsmart_bin_analysis" in p.parts:
            continue
        yield p


def import_hits(pattern: str) -> list[str]:
    rx = re.compile(pattern)
    hits = []
    for p in py_files():
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if rx.search(text):
            hits.append(str(p.relative_to(ROOT)))
    return hits


def main() -> int:
    # 1. Dependencies with zero references in the export tree (R-muse-02).
    for mod in ["pptx", "docxtpl", "latex2mathml", "cryptography",
                "requests", "joblib", "pydantic"]:
        hits = import_hits(rf"(?m)^\s*(from|import)\s+{mod}\b")
        check(f"dep-unused:{mod}", not hits, f"{len(hits)} hits" if hits else "no hits")

    # 2. torch_geometric is import-time required today, only via edges/ (R-muse-02).
    tg = import_hits(r"torch_geometric")
    check("torch_geometric-only-in-edges",
          tg and all("embeddings/edges" in h for h in tg), f"{len(tg)} files: {tg}")
    init = (LOGIC / "src/models/subnets/embeddings/__init__.py").read_text()
    check("embeddings-init-imports-edges", "from .edges import" in init)

    # 3. wandb only via metrics.py; jinja2 in export tree only via gui.py.
    check("wandb-only-metrics",
          import_hits(r"(?m)^\s*(from|import)\s+wandb\b")
          == ["logic/src/tracking/logging/modules/metrics.py"])
    check("jinja2-only-gui",
          import_hits(r"(?m)^\s*(from|import)\s+jinja2\b")
          == ["logic/src/tracking/logging/modules/gui.py"])

    # 4. networkx is used on the retained path (tsp.py) — must NOT be pruned.
    nx = import_hits(r"(?m)^\s*(from|import)\s+networkx\b")
    check("networkx-retained",
          any("travelling_salesman_problem/tsp.py" in h for h in nx), f"{len(nx)} files")

    # 5. Dead train dataclass fields: no readers outside logic/src/configs (R-muse-01).
    for field in ["route_improvement_epochs", "lr_route_improvement",
                  "efficiency_weight", "overflow_weight",
                  "accumulation_steps", "enable_scaler", "eval_only"]:
        hits = [str(p.relative_to(ROOT)) for p in py_files()
                if "src/configs" not in str(p)
                and re.search(rf"\b{field}\b", p.read_text(errors="ignore"))]
        check(f"dead-field:{field}", not hits, f"{hits}" if hits else "no readers")

    # 6. Live counter-examples that must stay OUT of R-claude-08.
    pbrs_hits = import_hits(r"use_pbrs")
    check("pbrs-is-read",
          bool(pbrs_hits)
          and "logic/src/pipeline/rl/common/base/module.py" in pbrs_hits)
    ce_hits = import_hits(r"checkpoint_encoder")
    check("checkpoint_encoder-is-read",
          "logic/src/utils/model/loader.py" in ce_hits)

    # 7. imitation_policy category points at a directory that does not exist (§4 item 7).
    check("models-policies-missing",
          not (LOGIC / "src/models/policies").exists())
    check("vector-policies-exist",
          (LOGIC / "src/policies/vector").is_dir())

    # 8. Packaging-script mechanisms on main (§4 items 1–5, Muse refinements).
    def show(rev_path: str) -> str:
        rel = rev_path.split(":", 1)[1]
        out = subprocess.run(["git", "show", rev_path], cwd=ROOT,
                             capture_output=True, text=True)
        if out.returncode != 0:
            # Fall back to the export-branch tree when main is unavailable.
            alt = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                                 capture_output=True, text=True)
            return alt.stdout if alt.returncode == 0 else ""
        return out.stdout

    prune = show("main:logic/package/prune_codebase.py")
    check("bulk-delete-empty-prefix",
          "should_delete = True" in prune and "else:" in prune,
          "empty yaml_prefixes deletes every yaml")
    helper = show("main:logic/package/cleanup_helper.py")
    check("parent-dir-kill",
          "to_delete.add(parent)" in helper,
          "matched file deletes its whole parent dir")
    check("single-line-init-comment",
          "new_lines.append(f\"# {line}  # AUTO-CLEANED\")" in helper,
          "continuation lines left uncommented")
    enums = show("main:logic/package/remove_enums.py")
    check("enums-regex-no-nesting",
          r"[^)]*?" in enums,
          "character class stops at first close paren")
    tracking = show("main:logic/package/remove_tracking.py")
    check("tracking-stubs-writers",
          "def update_policy_log_section(*args, **kwargs): pass" in tracking
          and "def send_final_output_to_gui(*args, **kwargs): pass" in tracking)

    print(f"\n{len(FAILURES)} failure(s).")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
