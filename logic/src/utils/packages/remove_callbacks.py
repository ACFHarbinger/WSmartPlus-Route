"""Script to clean up Callbacks components.

Deletes unselected callbacks based on user preferences.
If no preferences are specified, removes all callbacks.
"""

import argparse
import re
from pathlib import Path


def get_project_root() -> Path:
    """Find WSmart-Route root directory."""
    return Path(__file__).resolve().parents[4]


CALLBACKS = {
    "model_summary": {
        "class_name": "ModelSummaryCallback",
        "file_path": "logic/src/pipeline/callbacks/pytorch/model_summary.py",
        "import_line": "from .pytorch.model_summary import ModelSummaryCallback",
    },
    "reptile": {
        "class_name": "ReptileCallback",
        "file_path": "logic/src/pipeline/callbacks/pytorch/reptile.py",
        "import_line": "from .pytorch.reptile import ReptileCallback",
    },
    "speed_monitor": {
        "class_name": "SpeedMonitor",
        "file_path": "logic/src/pipeline/callbacks/pytorch/speed_monitor.py",
        "import_line": "from .pytorch.speed_monitor import SpeedMonitor",
    },
    "training_display": {
        "class_name": "TrainingDisplayCallback",
        "file_path": "logic/src/pipeline/callbacks/pytorch/training_display.py",
        "import_line": "from .pytorch.training_display import TrainingDisplayCallback",
    },
    "policy_summary": {
        "class_name": "PolicySummaryCallback",
        "file_path": "logic/src/pipeline/callbacks/simulation/policy_summary.py",
        "import_line": "from .simulation.policy_summary import PolicySummaryCallback",
    },
    "simulation_display": {
        "class_name": "SimulationDisplayCallback",
        "file_path": "logic/src/pipeline/callbacks/simulation/simulation_display.py",
        "import_line": "from .simulation.simulation_display import SimulationDisplayCallback",
    },
}


def clean_callbacks_init(init_path: Path, deleted_classes: list) -> None:
    """Clean up logic/src/pipeline/callbacks/__init__.py."""
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    for cls in deleted_classes:
        # Comment out import line
        pattern = rf"(?m)^(\s*from\s+\.\S+\s+import\s+.*{re.escape(cls)}.*)$"
        content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
        # Remove from __all__
        content = re.sub(rf'"{re.escape(cls)}",?\s*', "", content)
        content = re.sub(rf"'{re.escape(cls)}',?\s*", "", content)
    init_path.write_text(content)


def clean_trainer(trainer_path: Path, deleted_classes: list) -> None:
    """Clean up references in trainer.py if ModelSummaryCallback or TrainingDisplayCallback is deleted."""
    if not trainer_path.exists():
        return
    content = trainer_path.read_text(errors="ignore")

    # Clean imports from logic.src.pipeline.callbacks
    for cls in deleted_classes:
        pattern = r"(?m)^(\s*from\s+logic\.src\.pipeline\.callbacks\s+import\s+.*)$"
        match = re.search(pattern, content)
        if match:
            line = match.group(1)
            new_line = re.sub(rf"\b{re.escape(cls)}\b,?\s*", "", line)
            new_line = new_line.rstrip(", ")
            # If the import line becomes empty of classes, comment it out entirely
            if "import" in new_line and len(new_line.split("import")[-1].strip()) == 0:
                new_line = "# " + line
            # If it's a multi-line import, e.g. from logic.src.pipeline.callbacks import (\n ModelSummaryCallback,\n...)
            content = content.replace(line, new_line)

    # Specific cleanups for trainer.py usage blocks
    if "TrainingDisplayCallback" in deleted_classes:
        # Comment out the TrainingDisplayCallback checking/appending block
        block_pattern = r"(?m)([ \t]*)# Find if TrainingDisplayCallback exist.*?\n\s*callbacks\.append\(TrainingDisplayCallback\(\)\)\n"
        content = re.sub(
            block_pattern,
            lambda m: "\n".join(f"{m.group(1)}# {line.lstrip()}" for line in m.group(0).splitlines()) + "\n",
            content,
            flags=re.DOTALL,
        )

    if "ModelSummaryCallback" in deleted_classes:
        # Comment out the ModelSummaryCallback checking/appending block
        block_pattern = (
            r"(?m)([ \t]*)# Add custom model summary callback.*?\n\s*callbacks\.append\(ModelSummaryCallback\(\)\)\n"
        )
        content = re.sub(
            block_pattern,
            lambda m: "\n".join(f"{m.group(1)}# {line.lstrip()}" for line in m.group(0).splitlines()) + "\n",
            content,
            flags=re.DOTALL,
        )

    trainer_path.write_text(content)


def clean_config_yaml(config_yaml_path: Path) -> None:
    """Comment out TrainingDisplayCallback in config.yaml."""
    if not config_yaml_path.exists():
        return
    content = config_yaml_path.read_text(errors="ignore")

    # Comment out display callback section
    pattern = r"(?m)([ \t]*)- display:.*?\n\1        history_length: null"
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        block = match.group(0)
        commented_block = "\n".join(f"# {line}" for line in block.splitlines())
        content = content.replace(block, commented_block)

    config_yaml_path.write_text(content)


def main() -> None:
    """Perform callback cleanup based on user arguments."""
    parser = argparse.ArgumentParser(description="Remove callbacks except specified ones.")
    parser.add_argument(
        "keep",
        nargs="*",
        help="Names of callbacks to keep (comma-separated, space-separated, class names or short names)",
    )
    args = parser.parse_args()

    # Parse what to keep
    keep_list = []
    for item in args.keep:
        # split by commas
        for subitem in item.split(","):
            subitem = subitem.strip()
            if subitem:
                keep_list.append(subitem.lower())

    print("\n=== Removing Callbacks ===")
    print(f"Callbacks to keep: {keep_list or 'None (All will be deleted)'}")
    root = get_project_root()

    deleted_classes = []
    for name, info in CALLBACKS.items():
        # Check if matched in keep_list
        is_kept = False
        for k in keep_list:
            if k == name.lower() or k == info["class_name"].lower():
                is_kept = True
                break

        if not is_kept:
            file_path = root / info["file_path"]
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted callback file: {info['file_path']}")
            deleted_classes.append(info["class_name"])

    if deleted_classes:
        clean_callbacks_init(root / "logic/src/pipeline/callbacks/__init__.py", deleted_classes)
        clean_trainer(root / "logic/src/pipeline/rl/common/trainer.py", deleted_classes)
        if "TrainingDisplayCallback" in deleted_classes:
            clean_config_yaml(root / "logic/configs/config.yaml")

    print("=== Callback Cleanup Complete ===\n")


if __name__ == "__main__":
    main()
