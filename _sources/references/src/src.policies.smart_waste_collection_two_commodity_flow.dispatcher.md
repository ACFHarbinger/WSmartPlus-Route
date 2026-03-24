# {py:mod}`src.policies.smart_waste_collection_two_commodity_flow.dispatcher`

```{py:module} src.policies.smart_waste_collection_two_commodity_flow.dispatcher
```

```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.dispatcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_swc_tcf_optimizer <src.policies.smart_waste_collection_two_commodity_flow.dispatcher.run_swc_tcf_optimizer>`
  - ```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.dispatcher.run_swc_tcf_optimizer
    :summary:
    ```
````

### API

````{py:function} run_swc_tcf_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], values: typing.Dict[str, float], binsids: typing.List[int], must_go: typing.List[int], env: typing.Optional[gurobipy.Env] = None, number_vehicles: int = 1, time_limit: int = 60, optimizer: str = 'gurobi', seed: int = 42, max_iter_no_improv: int = 10)
:canonical: src.policies.smart_waste_collection_two_commodity_flow.dispatcher.run_swc_tcf_optimizer

```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.dispatcher.run_swc_tcf_optimizer
```
````
