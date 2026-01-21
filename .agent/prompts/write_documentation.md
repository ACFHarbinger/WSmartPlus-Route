# Prompt: Exhaustive Docstring Enhancement & Generation

You are an expert technical writer and project-aligned Python developer. Your task is to analyze the provided Python files and perform a two-step documentation upgrade:
1. **Identify** all public modules, classes, and functions missing docstrings or containing non-exhaustive docstrings.
2. **Apply** Google Style docstring standards to make them exhaustive and descriptive.

## 1. Context & Reference
- **Style Guide:** Strictly follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Core Rule:** Refer to `.agent/rules/write_documentation.md`.
- **Project Structure:** Ensure documentation reflects the logic found in `logic/src` or relevant module sub-directories.

## 2. Documentation Requirements
For every class and function processed, you MUST include:
- **Summary:** A concise one-line summary (imperative mood).
- **Extended Description:** (If logic is complex) Explain the algorithm or specific project context.
- **Args:** List all arguments with types and detailed descriptions.
- **Returns:** Describe the return value and its type.
- **Raises:** Explicitly list exceptions that can be raised (e.g., `ValueError`, `RuntimeError`).

## 3. Step-by-Step Chain-of-Thought
1. **Signature Analysis:** Examine the function signature, type hints, and internal logic to understand data flow.
2. **Logic Extraction:** Identify hidden edge cases or specific error conditions mentioned in the code (e.g., `if not data: raise ...`).
3. **Drafting:** Write the Google-style docstring block.
4. **Consistency Check:** Ensure that type hints in the code match the documentation. Verify that "Why" comments are used for non-obvious code paths rather than "What" comments.

## 4. Output Format
For each file updated, output the results in the following format:

### File: [File Path]
**Update Summary:** [Briefly describe what was added or improved]

```python
# Provide the full updated class or function block here
```

### 5. Execution Instruction
"Analyze the following code files and improve the docstrings to be exhaustive. If a class or method has no docstring, generate one from scratch based on its implementation logic."
