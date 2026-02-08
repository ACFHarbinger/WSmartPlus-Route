# {py:mod}`src.policies.selection.selection_combined`

```{py:module} src.policies.selection.selection_combined
```

```{autodoc2-docstring} src.policies.selection.selection_combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CombinedSelection <src.policies.selection.selection_combined.CombinedSelection>`
  - ```{autodoc2-docstring} src.policies.selection.selection_combined.CombinedSelection
    :summary:
    ```
````

### API

`````{py:class} CombinedSelection(strategies: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None, combined_strategies: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None, logic: str = 'or')
:canonical: src.policies.selection.selection_combined.CombinedSelection

Bases: {py:obj}`src.policies.selection.base.selection_strategy.MustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.selection.selection_combined.CombinedSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.selection_combined.CombinedSelection.__init__
```

````{py:method} select_bins(context: src.policies.selection.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.selection.selection_combined.CombinedSelection.select_bins

```{autodoc2-docstring} src.policies.selection.selection_combined.CombinedSelection.select_bins
```

````

````{py:method} _update_context(context: src.policies.selection.base.selection_context.SelectionContext, params: typing.Dict[str, typing.Any]) -> src.policies.selection.base.selection_context.SelectionContext
:canonical: src.policies.selection.selection_combined.CombinedSelection._update_context

```{autodoc2-docstring} src.policies.selection.selection_combined.CombinedSelection._update_context
```

````

`````
