# {py:mod}`src.policies.mandatory_selection.vectorized.base`

```{py:module} src.policies.mandatory_selection.vectorized.base
```

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedSelector <src.policies.mandatory_selection.vectorized.base.VectorizedSelector>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.base.VectorizedSelector
    :summary:
    ```
````

### API

`````{py:class} VectorizedSelector
:canonical: src.policies.mandatory_selection.vectorized.base.VectorizedSelector

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.base.VectorizedSelector
```

````{py:method} __init_subclass__(**kwargs: typing.Any) -> None
:canonical: src.policies.mandatory_selection.vectorized.base.VectorizedSelector.__init_subclass__
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.base.VectorizedSelector.__init_subclass__
```

````

````{py:method} select(fill_levels: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.mandatory_selection.vectorized.base.VectorizedSelector.select
:abstractmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.vectorized.base.VectorizedSelector.select
```

````

`````
