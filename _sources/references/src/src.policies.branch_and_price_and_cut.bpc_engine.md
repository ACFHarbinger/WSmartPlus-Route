# {py:mod}`src.policies.branch_and_price_and_cut.bpc_engine`

```{py:module} src.policies.branch_and_price_and_cut.bpc_engine
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_internal_bpc <src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc
    :summary:
    ```
````

### API

````{py:function} run_internal_bpc(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc
```
````
