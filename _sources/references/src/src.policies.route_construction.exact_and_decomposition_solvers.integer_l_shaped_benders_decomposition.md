# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params
src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine
src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.subproblem
src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem
src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd
src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.scenario
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_ils_bd <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.run_ils_bd>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.run_ils_bd
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.__all__>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.__all__
    :summary:
    ```
````

### API

````{py:function} run_ils_bd(dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams, mandatory_indices: typing.Set[int], vehicle_limit: int = 1, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.run_ils_bd

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.run_ils_bd
```
````

````{py:data} __all__
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.__all__
:value: >
   ['run_ils_bd', 'IntegerLShapedEngine', 'MasterProblem', 'RecourseEvaluator', 'ScenarioGenerator', 'I...

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.__all__
```

````
