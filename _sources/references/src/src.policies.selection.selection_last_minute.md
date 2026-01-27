# {py:mod}`src.policies.selection.selection_last_minute`

```{py:module} src.policies.selection.selection_last_minute
```

```{autodoc2-docstring} src.policies.selection.selection_last_minute
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LastMinuteSelection <src.policies.selection.selection_last_minute.LastMinuteSelection>`
  - ```{autodoc2-docstring} src.policies.selection.selection_last_minute.LastMinuteSelection
    :summary:
    ```
* - {py:obj}`LastMinuteAndPathSelection <src.policies.selection.selection_last_minute.LastMinuteAndPathSelection>`
  - ```{autodoc2-docstring} src.policies.selection.selection_last_minute.LastMinuteAndPathSelection
    :summary:
    ```
````

### API

`````{py:class} LastMinuteSelection
:canonical: src.policies.selection.selection_last_minute.LastMinuteSelection

Bases: {py:obj}`src.policies.must_go_selection.MustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.selection.selection_last_minute.LastMinuteSelection
```

````{py:method} select_bins(context: src.policies.must_go_selection.SelectionContext) -> typing.List[int]
:canonical: src.policies.selection.selection_last_minute.LastMinuteSelection.select_bins

````

`````

`````{py:class} LastMinuteAndPathSelection
:canonical: src.policies.selection.selection_last_minute.LastMinuteAndPathSelection

Bases: {py:obj}`src.policies.must_go_selection.MustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.selection.selection_last_minute.LastMinuteAndPathSelection
```

````{py:method} select_bins(context: src.policies.must_go_selection.SelectionContext) -> typing.List[int]
:canonical: src.policies.selection.selection_last_minute.LastMinuteAndPathSelection.select_bins

````

`````
