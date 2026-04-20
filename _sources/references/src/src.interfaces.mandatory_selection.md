# {py:mod}`src.interfaces.mandatory_selection`

```{py:module} src.interfaces.mandatory_selection
```

```{autodoc2-docstring} src.interfaces.mandatory_selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IMandatorySelectionStrategy <src.interfaces.mandatory_selection.IMandatorySelectionStrategy>`
  - ```{autodoc2-docstring} src.interfaces.mandatory_selection.IMandatorySelectionStrategy
    :summary:
    ```
````

### API

`````{py:class} IMandatorySelectionStrategy
:canonical: src.interfaces.mandatory_selection.IMandatorySelectionStrategy

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.mandatory_selection.IMandatorySelectionStrategy
```

````{py:method} select_bins(context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.SearchContext]
:canonical: src.interfaces.mandatory_selection.IMandatorySelectionStrategy.select_bins

```{autodoc2-docstring} src.interfaces.mandatory_selection.IMandatorySelectionStrategy.select_bins
```

````

`````
