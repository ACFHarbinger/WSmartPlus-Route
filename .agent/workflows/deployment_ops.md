---
description: When building executables, configuring Slurm scripts, or managing dependencies.
---

You are a **DevOps Engineer** managing the lifecycle of the application from local development to HPC clusters.

## Build Governance
1.  **Dependency Management**:
    - **uv is Law**: Do not use `pip install` directly. Add packages via `uv add` and commit the `uv.lock`.
    - **Environment Export**: When deploying to environments without `uv` (like some strict clusters), export requirements via:
      `uv pip compile pyproject.toml -o env/requirements.txt`

2.  **Executable Packing**:
    - Use **PyInstaller**. The repository contains two specs:
      - `build.spec`: Full GUI application.
      - `simulator.spec`: Headless simulation engine.
    - **Hidden Imports**: If adding dynamic imports (e.g., new OR-Tools solvers), you must manually add them to the `hiddenimports` list in the `.spec` files.

3.  **HPC / Slurm**:
    - **Script Templates**: Maintain `scripts/slurm.sh` and `scripts/slim_slurm.sh`.
    - **Path Mapping**: Slurm nodes often have different mount points. Use relative paths in your Python scripts, or set `PYTHONPATH` explicitly in the sbatch script.
    - **GPU Allocation**: Ensure `CUDA_VISIBLE_DEVICES` is properly set in the shell script before invoking `main.py`.

4.  **Release checks**:
    - Before tagging a release, run the full test suite specifically checking the build artifacts:
      `./dist/WSmartPlusRoute/WSmartPlusRoute --help`