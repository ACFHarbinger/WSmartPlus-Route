# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EngineNode <src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode
    :summary:
    ```
* - {py:obj}`ScenarioTreeExtensiveFormEngine <src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine
    :summary:
    ```
````

### API

`````{py:class} EngineNode
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode
```

````{py:attribute} id
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.id
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.id
```

````

````{py:attribute} day
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.day
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.day
```

````

````{py:attribute} probability
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.probability
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.probability
```

````

````{py:attribute} realization
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.realization
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.realization
```

````

````{py:attribute} parent_id
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.parent_id
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.parent_id
```

````

````{py:attribute} children_ids
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.children_ids
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode.children_ids
```

````

````{py:attribute} _sim_wastes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode._sim_wastes
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.EngineNode._sim_wastes
```

````

`````

`````{py:class} ScenarioTreeExtensiveFormEngine(tree: logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.tree.ScenarioTree, distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, waste_weight: float = 1.0, cost_weight: float = 1.0, overflow_penalty: float = 10.0, time_limit: float = 300.0, mip_gap: float = 0.05, use_mtz: bool = True, verbose: bool = False)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine.__init__
```

````{py:method} build_model()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine.build_model

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine.build_model
```

````

````{py:method} _add_inventory_constraints()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._add_inventory_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._add_inventory_constraints
```

````

````{py:method} _add_routing_constraints()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._add_routing_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._add_routing_constraints
```

````

````{py:method} _set_objective()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._set_objective

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._set_objective
```

````

````{py:method} solve() -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine.solve
```

````

````{py:method} _extract_route(n_idx: int) -> typing.List[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._extract_route

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._extract_route
```

````

````{py:method} _lazy_sec_callback(model: gurobipy.Model, where: int) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._lazy_sec_callback

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._lazy_sec_callback
```

````

````{py:method} _convert_simulation_tree(tree: typing.Any) -> typing.Dict[int, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._convert_simulation_tree

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._convert_simulation_tree
```

````

````{py:method} _get_engine_root() -> typing.Any
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._get_engine_root

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine.ScenarioTreeExtensiveFormEngine._get_engine_root
```

````

`````
