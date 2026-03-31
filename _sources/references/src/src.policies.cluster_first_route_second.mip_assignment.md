# {py:mod}`src.policies.cluster_first_route_second.mip_assignment`

```{py:module} src.policies.cluster_first_route_second.mip_assignment
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.mip_assignment
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_add_variables_and_objective <src.policies.cluster_first_route_second.mip_assignment._add_variables_and_objective>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.mip_assignment._add_variables_and_objective
    :summary:
    ```
* - {py:obj}`assign_exact_mip <src.policies.cluster_first_route_second.mip_assignment.assign_exact_mip>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.mip_assignment.assign_exact_mip
    :summary:
    ```
````

### API

````{py:function} _add_variables_and_objective(model: gurobipy.Model, must_go: typing.List[int], seeds: typing.List[int], costs: typing.Dict[typing.Tuple[int, int], float], wastes: typing.Dict[int, float], R: float, C: float, objective: str) -> typing.Dict[typing.Tuple[int, int], gurobipy.Var]
:canonical: src.policies.cluster_first_route_second.mip_assignment._add_variables_and_objective

```{autodoc2-docstring} src.policies.cluster_first_route_second.mip_assignment._add_variables_and_objective
```
````

````{py:function} assign_exact_mip(seeds: typing.List[int], must_go: typing.List[int], wastes: typing.Dict[int, float], capacity: float, R: float, C: float, distance_matrix: numpy.ndarray, time_limit: float = 60.0, objective: str = 'minimize_cost') -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.cluster_first_route_second.mip_assignment.assign_exact_mip

```{autodoc2-docstring} src.policies.cluster_first_route_second.mip_assignment.assign_exact_mip
```
````
