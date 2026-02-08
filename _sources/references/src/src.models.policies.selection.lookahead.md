# {py:mod}`src.models.policies.selection.lookahead`

```{py:module} src.models.policies.selection.lookahead
```

```{autodoc2-docstring} src.models.policies.selection.lookahead
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookaheadSelector <src.models.policies.selection.lookahead.LookaheadSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.lookahead.LookaheadSelector
    :summary:
    ```
````

### API

`````{py:class} LookaheadSelector(max_fill: float = 1.0)
:canonical: src.models.policies.selection.lookahead.LookaheadSelector

Bases: {py:obj}`src.models.policies.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.models.policies.selection.lookahead.LookaheadSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.selection.lookahead.LookaheadSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, max_fill: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.lookahead.LookaheadSelector.select

```{autodoc2-docstring} src.models.policies.selection.lookahead.LookaheadSelector.select
```

````

`````
