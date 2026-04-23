# {py:mod}`src.models.policies.selection.base`

```{py:module} src.models.policies.selection.base
```

```{autodoc2-docstring} src.models.policies.selection.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedSelector <src.models.policies.selection.base.VectorizedSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector
    :summary:
    ```
````

### API

`````{py:class} VectorizedSelector
:canonical: src.models.policies.selection.base.VectorizedSelector

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector
```

````{py:method} __init_subclass__(**kwargs: typing.Any) -> None
:canonical: src.models.policies.selection.base.VectorizedSelector.__init_subclass__
:classmethod:

```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector.__init_subclass__
```

````

````{py:method} select(fill_levels: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.policies.selection.base.VectorizedSelector.select
:abstractmethod:

```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector.select
```

````

`````
