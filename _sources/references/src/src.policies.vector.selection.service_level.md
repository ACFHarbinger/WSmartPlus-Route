# {py:mod}`src.policies.vector.selection.service_level`

```{py:module} src.policies.vector.selection.service_level
```

```{autodoc2-docstring} src.policies.vector.selection.service_level
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ServiceLevelSelector <src.policies.vector.selection.service_level.ServiceLevelSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.service_level.ServiceLevelSelector
    :summary:
    ```
````

### API

`````{py:class} ServiceLevelSelector(confidence_factor: float = 1.0, horizon_days: int = 1, **kwargs: typing.Any)
:canonical: src.policies.vector.selection.service_level.ServiceLevelSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.service_level.ServiceLevelSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.service_level.ServiceLevelSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, std_deviations: typing.Optional[torch.Tensor] = None, confidence_factor: typing.Optional[float] = None, horizon_days: typing.Optional[int] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.service_level.ServiceLevelSelector.select

```{autodoc2-docstring} src.policies.vector.selection.service_level.ServiceLevelSelector.select
```

````

`````
