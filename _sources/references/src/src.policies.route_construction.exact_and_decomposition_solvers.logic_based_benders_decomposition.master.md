# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBBDMasterProblem <src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem
    :summary:
    ```
````

### API

`````{py:class} LBBDMasterProblem(config: logic.src.configs.policies.lbbd.LBBDConfig, num_customers: int)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.__init__
```

````{py:method} build(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, stochastic_master: bool = False)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.build

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.build
```

````

````{py:method} _build_ev_model(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, horizon: int) -> typing.Any
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem._build_ev_model

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem._build_ev_model
```

````

````{py:method} _build_ef_model(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, horizon: int) -> typing.Any
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem._build_ef_model

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem._build_ef_model
```

````

````{py:method} add_nogood_cut(day: int, assigned_nodes: typing.List[int])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.add_nogood_cut

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.add_nogood_cut
```

````

````{py:method} add_optimality_cut(day: int, assigned_nodes: typing.List[int], distance: float)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.add_optimality_cut

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.add_optimality_cut
```

````

````{py:method} solve() -> typing.Tuple[typing.Dict[int, typing.List[int]], typing.Dict[int, float]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.master.LBBDMasterProblem.solve
```

````

`````
