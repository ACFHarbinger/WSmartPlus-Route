# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBBDEngine <src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine
    :summary:
    ```
````

### API

`````{py:class} LBBDEngine(config: logic.src.configs.policies.lbbd.LBBDConfig, distance_matrix: numpy.ndarray, tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, capacity: float = 1.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.lbbd_engine.LBBDEngine.solve
```

````

`````
