# {py:mod}`src.policies.smart_waste_collection_two_commodity_flow.hexaly`

```{py:module} src.policies.smart_waste_collection_two_commodity_flow.hexaly
```

```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.hexaly
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_run_hexaly_optimizer <src.policies.smart_waste_collection_two_commodity_flow.hexaly._run_hexaly_optimizer>`
  - ```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.hexaly._run_hexaly_optimizer
    :summary:
    ```
````

### API

````{py:function} _run_hexaly_optimizer(bins: numpy.typing.NDArray[numpy.float64], distancematrix: typing.List[typing.List[float]], values: typing.Dict[str, float], must_go: typing.List[int], number_vehicles: int = 1, time_limit: int = 60, seed: int = 42, max_iter_no_improv: int = 10, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.smart_waste_collection_two_commodity_flow.hexaly._run_hexaly_optimizer

```{autodoc2-docstring} src.policies.smart_waste_collection_two_commodity_flow.hexaly._run_hexaly_optimizer
```
````
