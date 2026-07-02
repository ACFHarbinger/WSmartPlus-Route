"""Script to clean up Hyperparameter Optimization (HPO) components.

Removes directories, configs, and config registrations associated with HPO.
"""

import re
import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Find WSmart-Route root directory."""
    return Path(__file__).resolve().parents[4]


def remove_path(path: Path) -> None:
    """Safely delete a file or directory."""
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        print(f"Deleted file: {path.relative_to(get_project_root())}")
    elif path.is_dir():
        shutil.rmtree(path)
        print(f"Deleted directory: {path.relative_to(get_project_root())}")


def clean_tasks_init(init_path: Path, class_names: list) -> None:
    """Comment out imports and remove from __all__ in tasks/__init__.py."""
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    for cls in class_names:
        pattern = rf"(?m)^(\s*from\s+\.\S+\s+import\s+.*{re.escape(cls)}.*)$"
        content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
        content = re.sub(rf'"{re.escape(cls)}",?\s*', "", content)
        content = re.sub(rf"'{re.escape(cls)}',?\s*", "", content)
    init_path.write_text(content)


def clean_configs_init(init_path: Path, class_names: list, field_names: list) -> None:
    """Clean up root configs/__init__.py imports and dataclass fields."""
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    for cls in class_names:
        # Remove from .tasks import line
        pattern = r"(?m)^(\s*from\s+\.tasks\s+import\s+.*)$"
        match = re.search(pattern, content)
        if match:
            line = match.group(1)
            new_line = re.sub(rf"\b{re.escape(cls)}\b,?\s*", "", line)
            new_line = new_line.rstrip(", ")
            if "import" in new_line and len(new_line.split("import")[-1].strip()) == 0:
                new_line = "# " + line
            content = content.replace(line, new_line)

        # Remove from __all__
        content = re.sub(rf'"{re.escape(cls)}",?\s*', "", content)
        content = re.sub(rf"'{re.escape(cls)}',?\s*", "", content)

    for field_name in field_names:
        # Remove config field declaration line
        field_pattern = rf"(?m)^\s*{re.escape(field_name)}\s*:\s*\S+\s*=\s*field\(default_factory=\S+\)\s*\n"
        content = re.sub(field_pattern, "", content)

    init_path.write_text(content)


def remove_constants_init_import(root: Path, module_name: str) -> None:
    """Comment out a wildcard import from logic/src/constants/__init__.py."""
    init_path = root / "logic/src/constants/__init__.py"
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    pattern = rf"(?m)^(from logic\.src\.constants\.{re.escape(module_name)} import \*.*)$"
    new_content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
    if new_content != content:
        print(f"Updated constants/__init__.py: commented out {module_name} import")
        init_path.write_text(new_content)


def clean_hydra_dispatch(dispatch_path: Path) -> None:
    """Remove HPO tasks and logic from hydra_dispatch.py."""
    if not dispatch_path.exists():
        return
    content = dispatch_path.read_text(errors="ignore")

    # Update run task triggers
    content = content.replace('"train", "meta_train", "hpo"', '"train", "meta_train"')
    content = content.replace('"train", "hpo"', '"train"')

    # Remove HPO simulation block
    hpo_sim_pattern = r"(?m)^\s*if task in \(\"hpo_sim\", \"sim_hpo\"\):.*?\n\s*return 0\.0\n"
    content = re.sub(hpo_sim_pattern, "", content, flags=re.DOTALL)

    # Clean up run_hpo call/import in features.train block
    content = content.replace(
        "from logic.src.pipeline.features.train import run_hpo, run_training",
        "from logic.src.pipeline.features.train import run_training",
    )
    hpo_check_pattern = r"(?m)^\s*if cfg\.hpo\.n_trials > 0:\s*\n\s*return run_hpo\(cfg\)\s*\n"
    content = re.sub(hpo_check_pattern, "", content)

    dispatch_path.write_text(content)


def main() -> None:
    """Perform cleanup of HPO components."""
    print("\n=== Removing Hyperparameter Optimization (HPO) Modules ===")
    root = get_project_root()

    # 1. Delete Directories
    remove_path(root / "logic/src/pipeline/rl/hpo")
    remove_path(root / "logic/src/pipeline/simulations/hpo")

    # 2. Delete Config and YAML files
    remove_path(root / "logic/src/configs/tasks/hpo_sim.py")
    remove_path(root / "logic/src/configs/tasks/hpo.py")
    remove_path(root / "logic/configs/tasks/hpo_sim.yaml")
    remove_path(root / "logic/configs/tasks/hpo.yaml")

    # 3. Clean up task registration in __init__.py files
    clean_tasks_init(root / "logic/src/configs/tasks/__init__.py", ["HPOConfig", "SimHPOConfig"])
    clean_configs_init(
        root / "logic/src/configs/__init__.py", ["HPOConfig", "SimHPOConfig"], ["hpo", "hpo_sim"]
    )

    # 4. Clean up hydra_dispatch.py
    clean_hydra_dispatch(root / "logic/controllers/hydra_dispatch.py")

    # 5. Remove HPO constants and update constants/__init__.py
    remove_path(root / "logic/src/constants/hpo.py")
    remove_constants_init_import(root, "hpo")

    print("=== HPO Cleanup Complete ===\n")


if __name__ == "__main__":
    main()
