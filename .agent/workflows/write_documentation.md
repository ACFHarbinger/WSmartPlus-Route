# Workflow: Documentation Update

**Trigger:** New feature implementation, code refactoring, or specific documentation debt reduction task.

## Phase 1: Analysis & Context
1.  **Identify Scope:** Determine which files were modified and require documentation updates.
2.  **Review Standards:** Briefly refresh memory on `.agent/rules/write_documentation.md` (Google Style).
3.  **Check Existing Docs:** Look at `logic/docs/` and parent `README.md` to see where new content fits.

## Phase 2: Python Docstring Update
For every modified Python file:
1.  **Module Level:** Ensure the file has a top-level docstring describing its module purpose.
2.  **Classes & Functions:**
    * **New Code:** Write full Google-style docstrings (Args, Returns, Raises).
    * **Modified Code:** Update existing docstrings to reflect changes in logic or signature.
3.  **Type Hinting:** Ensure Python type hints are present and match the `Args` description.

## Phase 3: Project-Level Documentation
1.  **README Update:**
    * If the usage of a script or module changed, update the nearest `README.md`.
    * If dependencies changed, update `DEPENDENCIES.md` or `env/requirements.txt`.
2.  **Sphinx Integration:**
    * If a new module was created, add it to `logic/docs/source/modules.rst` or the appropriate subdirectory index.
3.  **Changelog:**
    * Add a concise entry to `CHANGELOG.md` under the `[Unreleased]` section.

## Phase 4: Validation
1.  **Visual Check:** Read through the markdown and docstrings. Are they readable? Are there typos?
2.  **Consistency Check:** Do the docstring types match the function signature types?
3.  **Build Check (Optional):** If working extensively on Sphinx docs, run `make html` inside `logic/docs` to ensure no build errors.

## Phase 5: Final Review
1.  **Code Review:** Submit changes. Ensure reviewers check documentation as rigorously as code logic.
