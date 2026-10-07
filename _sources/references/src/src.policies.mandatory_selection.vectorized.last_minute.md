# {py:mod}`src.policies.mandatory_selection.vectorized.last_minute`

```{py:module} src.policies.mandatory_selection.vectorized.last_minute
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.last_minute
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LastMinuteSelector <src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector
    :summary:
    ```
````

### API

`````{py:class} LastMinuteSelector(threshold: float = 70.0)
:canonical: src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector

Bases: {py:obj}`src.policies.mandatory_selection.vectorized.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector.select

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.last_minute.LastMinuteSelector.select
```

````

`````
