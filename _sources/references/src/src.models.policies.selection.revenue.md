# {py:mod}`src.models.policies.selection.revenue`

```{py:module} src.models.policies.selection.revenue
```

```{autodoc2-docstring} src.models.policies.selection.revenue
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RevenueSelector <src.models.policies.selection.revenue.RevenueSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.revenue.RevenueSelector
    :summary:
    ```
````

### API

`````{py:class} RevenueSelector(revenue_kg: float = 1.0, bin_capacity: float = 1.0, threshold: float = 0.0)
:canonical: src.models.policies.selection.revenue.RevenueSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.revenue.RevenueSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.revenue.RevenueSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, revenue_kg: typing.Optional[float] = None, bin_capacity: typing.Optional[float] = None, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.revenue.RevenueSelector.select

```{autodoc2-docstring} src.models.policies.selection.revenue.RevenueSelector.select
```

````

`````
