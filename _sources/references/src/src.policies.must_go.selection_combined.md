# {py:mod}`src.policies.must_go.selection_combined`

```{py:module} src.policies.must_go.selection_combined
```

```{autodoc2-docstring} src.policies.must_go.selection_combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CombinedSelection <src.policies.must_go.selection_combined.CombinedSelection>`
  - ```{autodoc2-docstring} src.policies.must_go.selection_combined.CombinedSelection
    :summary:
    ```
````

### API

`````{py:class} CombinedSelection(strategies: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None, combined_strategies: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None, logic: str = 'or')
:canonical: src.policies.must_go.selection_combined.CombinedSelection

Bases: {py:obj}`logic.src.interfaces.must_go.MustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.must_go.selection_combined.CombinedSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.must_go.selection_combined.CombinedSelection.__init__
```

````{py:method} select_bins(context: src.policies.must_go.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.must_go.selection_combined.CombinedSelection.select_bins

```{autodoc2-docstring} src.policies.must_go.selection_combined.CombinedSelection.select_bins
```

````

````{py:method} _update_context(context: src.policies.must_go.base.selection_context.SelectionContext, params: typing.Dict[str, typing.Any]) -> src.policies.must_go.base.selection_context.SelectionContext
:canonical: src.policies.must_go.selection_combined.CombinedSelection._update_context

```{autodoc2-docstring} src.policies.must_go.selection_combined.CombinedSelection._update_context
```

````

`````
