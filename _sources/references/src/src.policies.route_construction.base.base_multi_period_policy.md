# {py:mod}`src.policies.route_construction.base.base_multi_period_policy`

```{py:module} src.policies.route_construction.base.base_multi_period_policy
```

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseMultiPeriodRoutingPolicy <src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy
    :summary:
    ```
````

### API

`````{py:class} BaseMultiPeriodRoutingPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy

Bases: {py:obj}`src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy.__init__
```

````{py:method} _get_scenario_tree(kwargs: typing.Dict[str, typing.Any]) -> logic.src.pipeline.simulations.bins.prediction.ScenarioTree
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._get_scenario_tree

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._get_scenario_tree
```

````

````{py:method} _predict_mandatory_nodes_for_horizon(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, initial_mandatory: typing.List[int]) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._predict_mandatory_nodes_for_horizon

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._predict_mandatory_nodes_for_horizon
```

````

````{py:method} _calculate_stockout_costs(inventory_sequence: numpy.ndarray) -> float
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._calculate_stockout_costs

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._calculate_stockout_costs
```

````

````{py:method} _run_multi_period_solver(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, capacity: float, revenue: float, cost_unit: float, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._run_multi_period_solver
:abstractmethod:

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy._run_multi_period_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, float, typing.Optional[logic.src.policies.context.search_context.SearchContext], typing.Optional[logic.src.policies.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy.execute
```

````

`````
