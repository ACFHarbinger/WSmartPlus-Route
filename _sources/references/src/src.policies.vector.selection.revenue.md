# {py:mod}`src.policies.vector.selection.revenue`

```{py:module} src.policies.vector.selection.revenue
```

```{autodoc2-docstring} src.policies.vector.selection.revenue
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RevenueSelector <src.policies.vector.selection.revenue.RevenueSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.revenue.RevenueSelector
    :summary:
    ```
````

### API

`````{py:class} RevenueSelector(revenue_kg: float = 1.0, bin_capacity: float = 1.0, threshold: float = 0.0)
:canonical: src.policies.vector.selection.revenue.RevenueSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.revenue.RevenueSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.revenue.RevenueSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, revenue_kg: typing.Optional[float] = None, bin_capacity: typing.Optional[float] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.revenue.RevenueSelector.select

```{autodoc2-docstring} src.policies.vector.selection.revenue.RevenueSelector.select
```

````

`````
