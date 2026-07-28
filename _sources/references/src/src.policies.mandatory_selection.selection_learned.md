# {py:mod}`src.policies.mandatory_selection.selection_learned`

```{py:module} src.policies.mandatory_selection.selection_learned
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LearnedSelection <src.policies.mandatory_selection.selection_learned.LearnedSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned.LearnedSelection
    :summary:
    ```
````

### API

`````{py:class} LearnedSelection(model_path: typing.Optional[str] = None, threshold: float = 0.5)
:canonical: src.policies.mandatory_selection.selection_learned.LearnedSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned.LearnedSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned.LearnedSelection.__init__
```

````{py:method} _load_model(path: str) -> None
:canonical: src.policies.mandatory_selection.selection_learned.LearnedSelection._load_model

```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned.LearnedSelection._load_model
```

````

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_learned.LearnedSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_learned.LearnedSelection.select_bins
```

````

`````
