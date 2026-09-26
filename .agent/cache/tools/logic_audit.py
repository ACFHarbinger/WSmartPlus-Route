"""Static structural audit of a Python package (adapted for logic/src from Image-Toolkit's gui_audit.py).

Usage: python logic_audit.py <root> [--long N] [--big N]

Reports: files over --big lines, functions over --long lines, classes with >=3 bases,
most-repeated def names across files, duck-typing sites (hasattr / getattr-with-default),
bare/silent excepts, hardcoded .cuda(), and module-level imports of heavy solver packages.
Every number is reproducible from the AST; no heuristics beyond what is named here.
"""
import ast
import collections
import os
import sys

HEAVY_IMPORTS = {"gurobipy", "hexaly", "localsolver", "ortools", "pyvrp", "alns", "torch_geometric", "lightning", "pytorch_lightning"}


def main():
    root = sys.argv[1]
    args = sys.argv[2:]
    long_n = int(args[args.index("--long") + 1]) if "--long" in args else 80
    big_n = int(args[args.index("--big") + 1]) if "--big" in args else 500

    files = []
    for dp, _dn, fn in os.walk(root):
        if "__pycache__" in dp:
            continue
        files.extend(os.path.join(dp, f) for f in fn if f.endswith(".py"))

    big_files, long_funcs, classes = [], [], []
    def_names = collections.defaultdict(set)
    hasattr_sites, getattr_default_sites = [], []
    bare_except, silent_except = [], []
    cuda_calls, heavy_module_imports = [], []
    total_loc = 0

    for path in files:
        rel = os.path.relpath(path, root)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception as e:  # noqa: BLE001
            print("PARSE FAIL", rel, e)
            continue
        n = src.count("\n") + 1
        total_loc += n
        if n > big_n:
            big_files.append((n, rel))

        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in HEAVY_IMPORTS:
                    heavy_module_imports.append((rel, node.lineno, name))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                classes.append((len(bases), node.name, rel, bases))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > long_n:
                    long_funcs.append((length, node.name, rel, node.lineno))
                def_names[node.name].add(rel)
            elif isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn == "hasattr":
                    hasattr_sites.append((rel, node.lineno))
                elif fn == "getattr" and len(node.args) == 3:
                    getattr_default_sites.append((rel, node.lineno))
                elif fn.endswith(".cuda"):
                    cuda_calls.append((rel, node.lineno))
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_except.append((rel, node.lineno))
                body = node.body
                if len(body) == 1 and (
                    isinstance(body[0], ast.Pass)
                    or (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant))
                ):
                    silent_except.append((rel, node.lineno))

    dunder = {"__init__", "__repr__", "__str__", "__len__", "__getitem__", "__call__", "__eq__", "__hash__", "__iter__", "__enter__", "__exit__"}
    repeated = sorted(
        ((len(v), k) for k, v in def_names.items() if k not in dunder and len(v) >= 5),
        reverse=True,
    )

    print(f"root={root} files={len(files)} loc={total_loc}")
    print(f"\n== files > {big_n} lines: {len(big_files)}")
    for n, rel in sorted(big_files, reverse=True):
        print(f"{n:6d}  {rel}")
    print(f"\n== functions > {long_n} lines: {len(long_funcs)}")
    for length, name, rel, line in sorted(long_funcs, reverse=True)[:60]:
        print(f"{length:5d}  {name:40s} {rel}:{line}")
    print(f"\n== classes with >= 3 bases: {sum(1 for c in classes if c[0] >= 3)} of {len(classes)}")
    for cnt, name, rel, bases in sorted((c for c in classes if c[0] >= 3), reverse=True):
        print(f"{cnt:3d}  {name:36s} {rel}  bases={bases}")
    print(f"\n== def names defined in >= 5 distinct files: {len(repeated)}")
    for cnt, name in repeated[:60]:
        print(f"{cnt:4d}  {name}")
    print(f"\n== hasattr( sites: {len(hasattr_sites)}   getattr(x, name, default) sites: {len(getattr_default_sites)}")
    by_file = collections.Counter(rel for rel, _ in hasattr_sites + getattr_default_sites)
    for rel, cnt in by_file.most_common(25):
        print(f"{cnt:4d}  {rel}")
    print(f"\n== bare except: {len(bare_except)}   except-with-only-pass/docstring: {len(silent_except)}")
    for rel, line in (bare_except + silent_except)[:60]:
        print(f"      {rel}:{line}")
    print(f"\n== hardcoded .cuda() calls: {len(cuda_calls)}")
    for rel, line in cuda_calls[:40]:
        print(f"      {rel}:{line}")
    print(f"\n== module-level heavy imports ({', '.join(sorted(HEAVY_IMPORTS))}): {len(heavy_module_imports)}")
    for rel, line, name in sorted(heavy_module_imports):
        print(f"      {name:16s} {rel}:{line}")


if __name__ == "__main__":
    main()
