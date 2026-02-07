#!/usr/bin/env python3
"""
Batch Docstring Adder Script

This script automatically adds docstrings to all remaining functions
that are missing them in the WSmart-Route codebase.
"""

import re
from pathlib import Path
from typing import List

# Files and their missing docstrings
MISSING_DOCSTRINGS = {
    "logic/src/models/policies/am.py": ["__init__"],
    "logic/src/models/policies/critic.py": ["__init__"],
    "logic/src/models/policies/utils.py": ["__init__", "get_edges_mask", "__init__"],
    "logic/src/models/policies/deep_decoder.py": ["__init__"],
    "logic/src/models/policies/pointer.py": ["__init__"],
    "logic/src/models/policies/symnco.py": ["__init__"],
    "logic/src/models/policies/temporal.py": ["__init__"],
    "logic/src/models/policies/classical/hybrid.py": ["__init__"],
    "logic/src/models/policies/classical/alns.py": ["__init__"],
    "logic/src/models/policies/classical/hgs.py": ["__init__"],
    "logic/src/models/embeddings/__init__.py": [
        "__init__",
        "__init__",
        "forward",
        "__init__",
        "forward",
    ],
    "logic/src/models/subnets/deep_decoder.py": ["__getitem__", "__init__"],
    "logic/src/pipeline/trainer.py": ["__init__"],
    "logic/src/pipeline/rl/features/post_processing.py": ["__init__"],
    "logic/src/pipeline/rl/hpo/optuna_hpo.py": ["__init__"],
    "logic/src/pipeline/rl/hpo/dehb.py": ["__init__", "get_incumbents"],
    "logic/src/pipeline/rl/meta/module.py": [
        "__init__",
        "validation_step",
        "test_step",
    ],
    "logic/src/pipeline/rl/meta/weight_optimizer.py": [
        "__init__",
        "propose_weights",
        "feedback",
        "update_histories",
        "prepare_meta_learning_batch",
        "meta_learning_step",
        "recommend_weights",
        "update_weights_internal",
        "get_current_weights",
    ],
    "logic/src/pipeline/rl/meta/td_learning.py": [
        "__init__",
        "state_dict",
        "load_state_dict",
    ],
    "logic/src/pipeline/rl/meta/hypernet_strategy.py": [
        "__init__",
        "get_current_weights",
    ],
}


DOCSTRING_TEMPLATES = {
    "__init__": '''"""
        Initialize {class_name}.

        Args:
            {args}
        """''',
    "forward": '''"""
        Forward pass.

        Args:
            {args}

        Returns:
            {return_type}
        """''',
    "default": '''"""
        {description}

        Args:
            {args}

        Returns:
            {return_type}
        """''',
}


def generate_docstring(func_name: str, class_name: str, args_list: List[str]) -> str:
    """Generate a docstring based on function signature."""
    # Clean up args
    clean_args = []
    for arg in args_list:
        if arg in ["self", "cls"]:
            continue
        # Remove type hints and defaults for docstring
        arg_name = arg.split(":")[0].split("=")[0].strip()
        clean_args.append(f"{arg_name}: Description for {arg_name}.")

    args_str = "\n            ".join(clean_args) if clean_args else "None."

    if func_name == "__init__":
        return f'''"""
        Initialize {class_name}.

        Args:
            {args_str}
        """'''
    elif func_name in ["forward", "validation_step", "test_step"]:
        return f'''"""
        {func_name.replace("_", " ").title()}.

        Args:
            {args_str}

        Returns:
            Computation result.
        """'''
    else:
        return f'''"""
        {func_name.replace("_", " ").title()}.

        Args:
            {args_str}
        """'''


def add_docstring_to_function(content: str, func_name: str, class_name: str = "") -> str:
    """Add docstring to a function if missing."""
    lines = content.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        result.append(line)

        # Check if this is a function definition
        if line.strip().startswith(f"def {func_name}("):
            # Check if next non-empty line is a docstring
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                result.append(lines[j])
                j += 1

            if j < len(lines):
                next_line = lines[j].strip()
                if not (next_line.startswith('"""') or next_line.startswith("'''")):
                    # No docstring, add one
                    # Extract args from function signature
                    func_def = line
                    k = i + 1
                    while k < len(lines) and "):" not in lines[k - 1]:
                        func_def += " " + lines[k].strip()
                        k += 1

                    # Simple arg extraction
                    args_match = re.search(r"\((.*?)\):", func_def, re.DOTALL)
                    if args_match:
                        args_str = args_match.group(1)
                        args_list = [a.strip() for a in args_str.split(",") if a.strip()]
                    else:
                        args_list = []

                    docstring = generate_docstring(func_name, class_name, args_list)

                    # Find indentation
                    indent = len(line) - len(line.lstrip()) + 4
                    docstring_lines = docstring.split("\n")
                    for ds_line in docstring_lines:
                        result.append(" " * indent + ds_line)

                    i = j - 1

        i += 1

    return "\n".join(result)


def process_file(file_path: str, missing_funcs: List[str]):
    """Process a single file and add missing docstrings."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    content = path.read_text()

    # Try to extract class name if any
    class_match = re.search(r"class\s+(\w+)", content)
    class_name = class_match.group(1) if class_match else "Class"

    modified = content
    for func in set(missing_funcs):  # Use set to avoid duplicates
        modified = add_docstring_to_function(modified, func, class_name)

    # Write back
    path.write_text(modified)
    print(f"✓ Processed {file_path}")


def main():
    """Run the batch docstring adder."""
    print("Starting batch docstring addition...")
    print(f"Processing {len(MISSING_DOCSTRINGS)} files...\n")

    for file_path, funcs in MISSING_DOCSTRINGS.items():
        try:
            process_file(file_path, funcs)
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")

    print("\nDone! Run check_docstrings.py to verify.")


if __name__ == "__main__":
    main()
