---
trigger: model_decision
description: When updating or creating python docstrings and documentation.
---

# Rule: Write Documentation

**Role:** Expert Technical Writer & Python Developer
**Objective:** Maintain high-quality, up-to-date, and standardized documentation across the codebase to ensure maintainability and developer velocity.

## 1. Core Principles
* **Completeness:** Every public module, class, and function MUST have a docstring.
* **Consistency:** Follow the **Google Python Style Guide** for all Python code.
* **Accuracy:** Documentation must strictly reflect the current state of the code. Code and docs must never diverge.
* **Clarity:** Use simple, direct language. Avoid jargon unless defined in the project glossary.

## 2. Python Docstring Standards
Adhere strictly to [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### 2.1 Function/Method Docstrings
Must include:
* **Summary:** A one-line summary of what the function does.
* **Description:** (Optional) Extended description for complex logic.
* **Args:** Detailed description of arguments, including types (if not fully typed in signature) and constraints.
* **Returns:** Description of the return value and type.
* **Raises:** List of all exceptions explicitly raised by the function.

**Example:**
```python
def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Trains the model for one epoch.

    Iterates through the data loader, computes loss, and performs a backward pass.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer used for weight updates.

    Returns:
        The average loss over the epoch.

    Raises:
        ValueError: If the input data is malformed.
    """
```

### 2.2 Class Docstrings
Must include:
* **Summary:** What the class represents.
* **Attributes:** Public attributes with types and descriptions.

## 3. Project Documentation (Markdown)
* **Format:** Project-level documentation must use GitHub Flavored Markdown (GFM).
* **Location and Hierarchy:**
    * **Root README:** The top-level `README.md` serves as the entry point, providing a high-level project overview and installation guides.
    * **Sub-directory READMEs:** Every major functional directory (e.g., `logic/src`, `gui/src`, `policy/src`) should contain its own `README.md` detailing the specific architecture and logic of that module.
* **Release Notes:** Significant changes, breaking updates, and bug fixes must be recorded in `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/) standard.
* **Dependencies:** Keep `DEPENDENCIES.md` synchronized with `env/requirements.txt` and `pyproject.toml`.

## 4. Sphinx Documentation
* **Documentation Engine:** The core technical documentation is generated via Sphinx, located in `logic/docs`.
* **API Reference:** When adding new Python modules, ensure they are exposed in the corresponding `.rst` files within `logic/docs/source/` to maintain the auto-generated API reference.
* **Build Integrity:** Documentation builds must be verified locally using `make html` before submission to ensure no broken cross-references or formatting errors.

## 5. Comments
* **Logic vs. Intent:** Avoid "what" comments (explaining what the code does). Prioritize "why" comments (explaining the rationale behind non-obvious logic or business rules).
* **Technical Debt:** Mark intentional temporary shortcuts or future improvements with `TODO(username): description`.
* **Cleanliness:** Remove all dead code, debug print statements, or large commented-out blocks before committing documentation updates.

## 6. Forbidden Practices
* **Stale Docs:** Never commit code changes without updating the associated documentation.
* **Trivial Docstrings:** Avoid redundant docstrings that simply repeat the function name (e.g., `def load_config(): """Loads the config."""`).
* **Format Mixing:** Do not mix different docstring styles; use Google Style exclusively.
