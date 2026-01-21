# WSmart-Route Dependency Policy

## Management
Dependencies are managed using `uv`. Core dependencies are listed in `pyproject.toml`.

## Security
We use `pip-audit` for daily vulnerability scanning in our CI pipeline.

## Update Policy
- **Security Updates**: Applied immediately.
- **Minor Updates**: Reviewed weekly via Dependabot.
- **Major Updates**: Reviewed quarterly.

## Standard Dependencies
- **DRL**: PyTorch 2.2.2, PyTorch Lightning
- **Config**: Hydra Core
- **OR Solvers**: Gurobi 11.0.3, Hexaly
- **GUI**: PySide6
