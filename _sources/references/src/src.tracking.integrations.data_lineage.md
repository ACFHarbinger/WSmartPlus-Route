# {py:mod}`src.tracking.integrations.data_lineage`

```{py:module} src.tracking.integrations.data_lineage
```

```{autodoc2-docstring} src.tracking.integrations.data_lineage
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataLineageCallback <src.tracking.integrations.data_lineage.DataLineageCallback>`
  - ```{autodoc2-docstring} src.tracking.integrations.data_lineage.DataLineageCallback
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_bins_to_tensor_dict <src.tracking.integrations.data_lineage._bins_to_tensor_dict>`
  - ```{autodoc2-docstring} src.tracking.integrations.data_lineage._bins_to_tensor_dict
    :summary:
    ```
````

### API

`````{py:class} DataLineageCallback(policy_name: str, sample_id: int, log_freq: int = 1)
:canonical: src.tracking.integrations.data_lineage.DataLineageCallback

```{autodoc2-docstring} src.tracking.integrations.data_lineage.DataLineageCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.data_lineage.DataLineageCallback.__init__
```

````{py:method} on_simulation_start(context: typing.Any) -> None
:canonical: src.tracking.integrations.data_lineage.DataLineageCallback.on_simulation_start

```{autodoc2-docstring} src.tracking.integrations.data_lineage.DataLineageCallback.on_simulation_start
```

````

````{py:method} on_step_end(day_context: typing.Any, day: int) -> None
:canonical: src.tracking.integrations.data_lineage.DataLineageCallback.on_step_end

```{autodoc2-docstring} src.tracking.integrations.data_lineage.DataLineageCallback.on_step_end
```

````

`````

````{py:function} _bins_to_tensor_dict(bins: typing.Any) -> typing.Dict[str, torch.Tensor]
:canonical: src.tracking.integrations.data_lineage._bins_to_tensor_dict

```{autodoc2-docstring} src.tracking.integrations.data_lineage._bins_to_tensor_dict
```
````
