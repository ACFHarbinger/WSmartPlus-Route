# {py:mod}`src.policies.other.must_go.selection_learned`

```{py:module} src.policies.other.must_go.selection_learned
```

```{autodoc2-docstring} src.policies.other.must_go.selection_learned
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LearnedSelection <src.policies.other.must_go.selection_learned.LearnedSelection>`
  - ```{autodoc2-docstring} src.policies.other.must_go.selection_learned.LearnedSelection
    :summary:
    ```
````

### API

`````{py:class} LearnedSelection(model_path: typing.Optional[str] = None, threshold: float = 0.5)
:canonical: src.policies.other.must_go.selection_learned.LearnedSelection

Bases: {py:obj}`logic.src.interfaces.must_go.IMustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.other.must_go.selection_learned.LearnedSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.must_go.selection_learned.LearnedSelection.__init__
```

````{py:method} _load_model(path: str) -> None
:canonical: src.policies.other.must_go.selection_learned.LearnedSelection._load_model

```{autodoc2-docstring} src.policies.other.must_go.selection_learned.LearnedSelection._load_model
```

````

````{py:method} select_bins(context: logic.src.policies.other.must_go.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_learned.LearnedSelection.select_bins

```{autodoc2-docstring} src.policies.other.must_go.selection_learned.LearnedSelection.select_bins
```

````

`````
