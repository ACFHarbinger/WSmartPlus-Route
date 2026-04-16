# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SWCTCFPolicy <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy
    :summary:
    ```
````

### API

`````{py:class} SWCTCFPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.SWCTCFConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.policies.context.search_context.SearchContext], typing.Optional[logic.src.policies.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf.SWCTCFPolicy.execute
```

````

`````
