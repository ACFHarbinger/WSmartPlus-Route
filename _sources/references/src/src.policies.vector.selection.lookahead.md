# {py:mod}`src.policies.vector.selection.lookahead`

```{py:module} src.policies.vector.selection.lookahead
```

```{autodoc2-docstring} src.policies.vector.selection.lookahead
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookaheadSelector <src.policies.vector.selection.lookahead.LookaheadSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.lookahead.LookaheadSelector
    :summary:
    ```
````

### API

`````{py:class} LookaheadSelector(current_collection_day: int = 0, **kwargs: typing.Any)
:canonical: src.policies.vector.selection.lookahead.LookaheadSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.lookahead.LookaheadSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.lookahead.LookaheadSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, current_collection_day: typing.Optional[typing.Union[int, torch.Tensor]] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.lookahead.LookaheadSelector.select

```{autodoc2-docstring} src.policies.vector.selection.lookahead.LookaheadSelector.select
```

````

`````
