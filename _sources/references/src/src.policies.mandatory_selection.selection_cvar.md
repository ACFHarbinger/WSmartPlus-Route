# {py:mod}`src.policies.mandatory_selection.selection_cvar`

```{py:module} src.policies.mandatory_selection.selection_cvar
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_cvar
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVaRSelection <src.policies.mandatory_selection.selection_cvar.CVaRSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_cvar.CVaRSelection
    :summary:
    ```
````

### API

`````{py:class} CVaRSelection
:canonical: src.policies.mandatory_selection.selection_cvar.CVaRSelection

Bases: {py:obj}`logic.src.interfaces.mandatory.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_cvar.CVaRSelection
```

````{py:method} select_bins(context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_cvar.CVaRSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_cvar.CVaRSelection.select_bins
```

````

`````
