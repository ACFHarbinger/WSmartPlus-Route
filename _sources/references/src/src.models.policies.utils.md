# {py:mod}`src.models.policies.utils`

```{py:module} src.models.policies.utils
```

```{autodoc2-docstring} src.models.policies.utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorDictStateWrapper <src.models.policies.utils.TensorDictStateWrapper>`
  - ```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper
    :summary:
    ```
* - {py:obj}`DummyProblem <src.models.policies.utils.DummyProblem>`
  - ```{autodoc2-docstring} src.models.policies.utils.DummyProblem
    :summary:
    ```
````

### API

`````{py:class} TensorDictStateWrapper(td: tensordict.TensorDict, problem_name: str = 'vrpp', env=None)
:canonical: src.models.policies.utils.TensorDictStateWrapper

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.__init__
```

````{py:method} get_mask() -> typing.Optional[torch.Tensor]
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_mask

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_mask
```

````

````{py:method} get_edges_mask() -> typing.Optional[torch.Tensor]
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_edges_mask

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_edges_mask
```

````

````{py:method} get_current_node() -> torch.Tensor
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_current_node

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_current_node
```

````

````{py:method} get_current_profit() -> torch.Tensor
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_current_profit

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_current_profit
```

````

````{py:method} get_current_efficiency() -> torch.Tensor
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_current_efficiency

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_current_efficiency
```

````

````{py:method} get_remaining_overflows() -> torch.Tensor
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_remaining_overflows

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_remaining_overflows
```

````

````{py:method} update(action: torch.Tensor) -> src.models.policies.utils.TensorDictStateWrapper
:canonical: src.models.policies.utils.TensorDictStateWrapper.update

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.update
```

````

````{py:method} all_finished() -> bool
:canonical: src.models.policies.utils.TensorDictStateWrapper.all_finished

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.all_finished
```

````

````{py:method} get_finished() -> torch.Tensor
:canonical: src.models.policies.utils.TensorDictStateWrapper.get_finished

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.get_finished
```

````

````{py:method} __getitem__(key)
:canonical: src.models.policies.utils.TensorDictStateWrapper.__getitem__

```{autodoc2-docstring} src.models.policies.utils.TensorDictStateWrapper.__getitem__
```

````

`````

````{py:class} DummyProblem(name: str)
:canonical: src.models.policies.utils.DummyProblem

```{autodoc2-docstring} src.models.policies.utils.DummyProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.utils.DummyProblem.__init__
```

````
