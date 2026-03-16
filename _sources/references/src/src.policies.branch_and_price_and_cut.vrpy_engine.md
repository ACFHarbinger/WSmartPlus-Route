# {py:mod}`src.policies.branch_and_price_and_cut.vrpy_engine`

```{py:module} src.policies.branch_and_price_and_cut.vrpy_engine
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpy_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bpc_vrpy <src.policies.branch_and_price_and_cut.vrpy_engine.run_bpc_vrpy>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpy_engine.run_bpc_vrpy
    :summary:
    ```
````

### API

````{py:function} run_bpc_vrpy(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_price_and_cut.vrpy_engine.run_bpc_vrpy

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpy_engine.run_bpc_vrpy
```
````
