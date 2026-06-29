# {py:mod}`src.policies.vector.selection.base`

```{py:module} src.policies.vector.selection.base
```

```{autodoc2-docstring} src.policies.vector.selection.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedSelector <src.policies.vector.selection.base.VectorizedSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.base.VectorizedSelector
    :summary:
    ```
````

### API

`````{py:class} VectorizedSelector
:canonical: src.policies.vector.selection.base.VectorizedSelector

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.vector.selection.base.VectorizedSelector
```

````{py:method} __init_subclass__(**kwargs: typing.Any) -> None
:canonical: src.policies.vector.selection.base.VectorizedSelector.__init_subclass__
:classmethod:

```{autodoc2-docstring} src.policies.vector.selection.base.VectorizedSelector.__init_subclass__
```

````

````{py:method} select(fill_levels: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.base.VectorizedSelector.select
:abstractmethod:

```{autodoc2-docstring} src.policies.vector.selection.base.VectorizedSelector.select
```

````

`````
