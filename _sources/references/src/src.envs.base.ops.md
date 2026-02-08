# {py:mod}`src.envs.base.ops`

```{py:module} src.envs.base.ops
```

```{autodoc2-docstring} src.envs.base.ops
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OpsMixin <src.envs.base.ops.OpsMixin>`
  - ```{autodoc2-docstring} src.envs.base.ops.OpsMixin
    :summary:
    ```
````

### API

`````{py:class} OpsMixin
:canonical: src.envs.base.ops.OpsMixin

```{autodoc2-docstring} src.envs.base.ops.OpsMixin
```

````{py:method} _make_spec(generator: typing.Optional[typing.Any] = None) -> None
:canonical: src.envs.base.ops.OpsMixin._make_spec

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._make_spec
```

````

````{py:method} step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin.step

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.step
```

````

````{py:method} reset(td: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin.reset

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.reset
```

````

````{py:method} _reset(td: typing.Optional[tensordict.TensorDict] = None, batch_size: typing.Optional[int] = None) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin._reset

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._reset
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin._step

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._step
```

````

````{py:method} make_state(nodes: tensordict.TensorDict, edges: typing.Optional[torch.Tensor] = None, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.base.ops.OpsMixin.make_state

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.make_state
```

````

````{py:method} get_costs(td: tensordict.TensorDict, pi: torch.Tensor, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Any
:canonical: src.envs.base.ops.OpsMixin.get_costs

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.get_costs
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin._step_instance

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin._check_done

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._check_done
```

````

````{py:method} get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin.get_reward

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.get_reward
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.ops.OpsMixin._reset_instance
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._reset_instance
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin._get_reward
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._get_reward
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin._get_action_mask
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._get_action_mask
```

````

````{py:method} _set_seed(seed: typing.Optional[int])
:canonical: src.envs.base.ops.OpsMixin._set_seed

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._set_seed
```

````

`````
