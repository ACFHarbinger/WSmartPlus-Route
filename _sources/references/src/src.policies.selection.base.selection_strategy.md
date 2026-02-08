# {py:mod}`src.policies.selection.base.selection_strategy`

```{py:module} src.policies.selection.base.selection_strategy
```

```{autodoc2-docstring} src.policies.selection.base.selection_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MustGoSelectionStrategy <src.policies.selection.base.selection_strategy.MustGoSelectionStrategy>`
  - ```{autodoc2-docstring} src.policies.selection.base.selection_strategy.MustGoSelectionStrategy
    :summary:
    ```
````

### API

`````{py:class} MustGoSelectionStrategy
:canonical: src.policies.selection.base.selection_strategy.MustGoSelectionStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.selection.base.selection_strategy.MustGoSelectionStrategy
```

````{py:method} select_bins(context: src.policies.selection.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.selection.base.selection_strategy.MustGoSelectionStrategy.select_bins
:abstractmethod:

```{autodoc2-docstring} src.policies.selection.base.selection_strategy.MustGoSelectionStrategy.select_bins
```

````

`````
