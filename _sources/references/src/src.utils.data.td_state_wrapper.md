# {py:mod}`src.utils.data.td_state_wrapper`

```{py:module} src.utils.data.td_state_wrapper
```

```{autodoc2-docstring} src.utils.data.td_state_wrapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorDictStateWrapper <src.utils.data.td_state_wrapper.TensorDictStateWrapper>`
  - ```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper
    :summary:
    ```
````

### API

`````{py:class} TensorDictStateWrapper(td: tensordict.TensorDict, problem_name: str = 'vrpp', env=None)
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.__init__
```

````{py:method} get_mask() -> typing.Optional[torch.Tensor]
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_mask

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_mask
```

````

````{py:method} get_edges_mask() -> typing.Optional[torch.Tensor]
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_edges_mask

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_edges_mask
```

````

````{py:method} get_current_node() -> torch.Tensor
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_node

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_node
```

````

````{py:method} get_current_profit() -> torch.Tensor
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_profit

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_profit
```

````

````{py:method} get_current_efficiency() -> torch.Tensor
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_efficiency

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_current_efficiency
```

````

````{py:method} get_remaining_overflows() -> torch.Tensor
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_remaining_overflows

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_remaining_overflows
```

````

````{py:method} update(action: torch.Tensor) -> src.utils.data.td_state_wrapper.TensorDictStateWrapper
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.update

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.update
```

````

````{py:method} all_finished() -> bool
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.all_finished

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.all_finished
```

````

````{py:method} get_finished() -> torch.Tensor
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_finished

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.get_finished
```

````

````{py:method} __getitem__(key)
:canonical: src.utils.data.td_state_wrapper.TensorDictStateWrapper.__getitem__

```{autodoc2-docstring} src.utils.data.td_state_wrapper.TensorDictStateWrapper.__getitem__
```

````

`````
