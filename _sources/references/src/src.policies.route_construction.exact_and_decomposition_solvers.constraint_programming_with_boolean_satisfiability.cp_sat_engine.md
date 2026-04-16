# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CPSATEngine <src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine
    :summary:
    ```
````

### API

`````{py:class} CPSATEngine(config: logic.src.configs.policies.cp_sat.CPSATConfig, distance_matrix: numpy.ndarray, initial_wastes: typing.Dict[int, float], capacity: float = 1.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine.solve
```

````

````{py:method} _init_variables(model: ortools.sat.python.cp_model.CpModel) -> typing.Dict[str, typing.Dict]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._init_variables

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._init_variables
```

````

````{py:method} _add_routing_constraints(model: ortools.sat.python.cp_model.CpModel, vars_dict: typing.Dict[str, typing.Dict])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_routing_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_routing_constraints
```

````

````{py:method} _add_inventory_constraints(model: ortools.sat.python.cp_model.CpModel, vars_dict: typing.Dict[str, typing.Dict])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_inventory_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_inventory_constraints
```

````

````{py:method} _add_objective(model: ortools.sat.python.cp_model.CpModel, vars_dict: typing.Dict[str, typing.Dict])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_objective

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._add_objective
```

````

````{py:method} _extract_route(solver: ortools.sat.python.cp_model.CpSolver, x: typing.Dict) -> typing.List[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._extract_route

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.cp_sat_engine.CPSATEngine._extract_route
```

````

`````
