# {py:mod}`src.policies.mandatory_selection.selection_service_level`

```{py:module} src.policies.mandatory_selection.selection_service_level
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_service_level
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ServiceLevelSelection <src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection
    :summary:
    ```
````

### API

`````{py:class} ServiceLevelSelection
:canonical: src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection
```

````{py:method} select_bins(context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_service_level.ServiceLevelSelection.select_bins
```

````

`````
