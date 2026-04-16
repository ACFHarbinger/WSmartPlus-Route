# {py:mod}`src.interfaces.mandatory`

```{py:module} src.interfaces.mandatory
```

```{autodoc2-docstring} src.interfaces.mandatory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IMandatorySelectionStrategy <src.interfaces.mandatory.IMandatorySelectionStrategy>`
  - ```{autodoc2-docstring} src.interfaces.mandatory.IMandatorySelectionStrategy
    :summary:
    ```
````

### API

`````{py:class} IMandatorySelectionStrategy
:canonical: src.interfaces.mandatory.IMandatorySelectionStrategy

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.mandatory.IMandatorySelectionStrategy
```

````{py:method} select_bins(context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.SearchContext]
:canonical: src.interfaces.mandatory.IMandatorySelectionStrategy.select_bins

```{autodoc2-docstring} src.interfaces.mandatory.IMandatorySelectionStrategy.select_bins
```

````

`````
