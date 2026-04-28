# {py:mod}`src.policies.helpers.solvers_and_matheuristics.search.column_generation`

```{py:module} src.policies.helpers.solvers_and_matheuristics.search.column_generation
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.column_generation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`column_generation_loop <src.policies.helpers.solvers_and_matheuristics.search.column_generation.column_generation_loop>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.column_generation.column_generation_loop
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.search.column_generation.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.column_generation.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.search.column_generation.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.column_generation.logger
```

````

````{py:function} column_generation_loop(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.pricing.RCSPPSolver, cut_engine: logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.AnyBranchingConstraint]], max_cg_iterations: int, max_cuts: int, time_limit: float, start_time: float, max_routes_per_pricing: int = 5, vehicle_limit: typing.Optional[int] = None, optimality_gap: float = 0.0001, early_termination_gap: float = 0.001, parent_basis: typing.Optional[typing.Any] = None, incumbent_value: float = -float('inf'), node_depth: int = 0, rc_tolerance: float = 1e-05, cut_orthogonality_threshold: float = 0.8, exact_mode: bool = False, cg_at_root_only: bool = False, branching_strategy: str = 'divergence', rcspp_timeout: float = 30.0) -> typing.Tuple[float, typing.Dict[int, float], typing.Optional[typing.Any], bool]
:canonical: src.policies.helpers.solvers_and_matheuristics.search.column_generation.column_generation_loop

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.column_generation.column_generation_loop
```
````
