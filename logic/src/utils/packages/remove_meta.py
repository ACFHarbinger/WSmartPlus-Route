"""Script to clean up Meta Learning components.

Removes directories, configs, and config registrations associated with Meta Learning.
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


def clean_rl_init(rl_init_path: Path) -> None:
    """Remove Meta-RL and HRL references from logic/src/pipeline/rl/__init__.py."""
    if not rl_init_path.exists():
        return
    content = rl_init_path.read_text(errors="ignore")

    # Remove HRLModule and MetaRLModule from from logic.src.pipeline.rl.core import line
    import_pattern = r"from logic\.src\.pipeline\.rl\.core import \(\s*.*?\s*\)"
    match = re.search(import_pattern, content, flags=re.DOTALL)
    if match:
        block = match.group(0)
        new_block = block.replace("HRLModule,", "").replace("MetaRLModule,", "")
        content = content.replace(block, new_block)

    # Remove registry mappings
    content = re.sub(r'^\s*"meta_rl"\s*:\s*MetaRLModule,\s*\n', "", content, flags=re.MULTILINE)
    content = re.sub(r'^\s*"hrl"\s*:\s*HRLModule,\s*\n', "", content, flags=re.MULTILINE)

    # Remove from __all__
    content = content.replace('"MetaRLModule",', "").replace('"HRLModule",', "")

    rl_init_path.write_text(content)


def clean_rl_core_init(core_init_path: Path) -> None:
    """Remove meta strategy imports from logic/src/pipeline/rl/core/__init__.py."""
    if not core_init_path.exists():
        return
    content = core_init_path.read_text(errors="ignore")

    # Comment out HRLModule and MetaRLModule imports
    content = re.sub(
        r"(?m)^(\s*from logic\.src\.pipeline\.rl\.meta\.hrl import HRLModule)$",
        r"# \1  # AUTO-REMOVED",
        content,
    )
    content = re.sub(
        r"(?m)^(\s*from logic\.src\.pipeline\.rl\.meta\.module import MetaRLModule)$",
        r"# \1  # AUTO-REMOVED",
        content,
    )

    # Remove from __all__
    content = content.replace('"MetaRLModule",', "").replace('"HRLModule",', "")

    core_init_path.write_text(content)


def clean_builder(builder_path: Path) -> None:
    """Remove meta and HRL references from builder.py."""
    if not builder_path.exists():
        return
    content = builder_path.read_text(errors="ignore")

    # Remove MetaRLModule from import
    content = content.replace(", MetaRLModule", "")

    # Remove MetaRLModule instantiation block
    meta_block = r"if getattr\(cfg\.meta_rl, \"use_meta\", False\):.*?model = MetaRLModule\(.*?\n\s*\)\s*\n"
    content = re.sub(meta_block, "", content, flags=re.DOTALL)

    # Remove MetaRLModule parameter logging block
    logging_block = (
        r"if getattr\(cfg\.meta_rl, \"use_meta\", False\):.*?params\[\"model\.meta_lr\"\] = .*?\n"
    )
    content = re.sub(logging_block, "", content, flags=re.DOTALL)

    builder_path.write_text(content)


def clean_hydra_dispatch(dispatch_path: Path) -> None:
    """Remove meta tasks from hydra_dispatch.py."""
    if not dispatch_path.exists():
        return
    content = dispatch_path.read_text(errors="ignore")

    # Remove meta_train task from runners
    content = content.replace('"train", "meta_train"', '"train"')

    dispatch_path.write_text(content)


def main() -> None:
    """Perform cleanup of Meta Learning components."""
    print("\n=== Removing Meta Learning Modules ===")
    root = get_project_root()

    # 1. Delete Directories
    remove_path(root / "logic/src/models/meta")
    remove_path(root / "logic/src/pipeline/rl/meta")

    # 2. Delete Config and YAML files
    remove_path(root / "logic/src/configs/tasks/meta_rl.py")
    remove_path(root / "logic/configs/tasks/meta_train.yaml")

    # 3. Delete hrl.py model factory builder
    remove_path(root / "logic/src/pipeline/features/train/model_factory/hrl.py")

    # 4. Clean up configs and registries
    clean_tasks_init(root / "logic/src/configs/tasks/__init__.py", ["MetaRLConfig"])
    clean_configs_init(root / "logic/src/configs/__init__.py", ["MetaRLConfig"], ["meta_rl"])
    clean_rl_init(root / "logic/src/pipeline/rl/__init__.py")
    clean_rl_core_init(root / "logic/src/pipeline/rl/core/__init__.py")
    clean_builder(root / "logic/src/pipeline/features/train/model_factory/builder.py")
    clean_hydra_dispatch(root / "logic/hydra_dispatch.py")

    print("=== Meta Learning Cleanup Complete ===\n")


if __name__ == "__main__":
    main()
