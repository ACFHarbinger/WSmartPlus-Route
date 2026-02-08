# {py:mod}`src.models.policies.selection.service_level`

```{py:module} src.models.policies.selection.service_level
```

```{autodoc2-docstring} src.models.policies.selection.service_level
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ServiceLevelSelector <src.models.policies.selection.service_level.ServiceLevelSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.service_level.ServiceLevelSelector
    :summary:
    ```
````

### API

`````{py:class} ServiceLevelSelector(confidence_factor: float = 1.0, max_fill: float = 1.0)
:canonical: src.models.policies.selection.service_level.ServiceLevelSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.service_level.ServiceLevelSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.service_level.ServiceLevelSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, std_deviations: typing.Optional[torch.Tensor] = None, confidence_factor: typing.Optional[float] = None, max_fill: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.service_level.ServiceLevelSelector.select

```{autodoc2-docstring} src.models.policies.selection.service_level.ServiceLevelSelector.select
```

````

`````
