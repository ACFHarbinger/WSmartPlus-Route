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

````{py:method} step(tensordict: tensordict.TensorDictBase) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin.step

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.step
```

````

````{py:method} reset(tensordict: typing.Optional[tensordict.TensorDictBase] = None, **kwargs) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin.reset

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.reset
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDictBase] = None, **kwargs) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin._reset

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._reset
```

````

````{py:method} _step(tensordict: tensordict.TensorDictBase) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin._step

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._step
```

````

````{py:method} make_state(nodes: tensordict.TensorDictBase, edges: typing.Optional[torch.Tensor] = None, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.base.ops.OpsMixin.make_state

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.make_state
```

````

````{py:method} get_costs(tensordict: tensordict.TensorDictBase, pi: torch.Tensor, cost_weights: typing.Optional[torch.Tensor] = None, dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Any
:canonical: src.envs.base.ops.OpsMixin.get_costs

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.get_costs
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDictBase) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin._step_instance

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._step_instance
```

````

````{py:method} _check_done(tensordict: tensordict.TensorDictBase) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin._check_done

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._check_done
```

````

````{py:method} get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin.get_reward

```{autodoc2-docstring} src.envs.base.ops.OpsMixin.get_reward
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDictBase) -> tensordict.TensorDictBase
:canonical: src.envs.base.ops.OpsMixin._reset_instance
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._reset_instance
```

````

````{py:method} _get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.ops.OpsMixin._get_reward
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ops.OpsMixin._get_reward
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDictBase) -> torch.Tensor
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
