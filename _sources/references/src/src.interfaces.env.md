# {py:mod}`src.interfaces.env`

```{py:module} src.interfaces.env
```

```{autodoc2-docstring} src.interfaces.env
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IEnv <src.interfaces.env.IEnv>`
  - ```{autodoc2-docstring} src.interfaces.env.IEnv
    :summary:
    ```
````

### API

`````{py:class} IEnv
:canonical: src.interfaces.env.IEnv

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.env.IEnv
```

````{py:attribute} name
:canonical: src.interfaces.env.IEnv.name
:type: str
:value: >
   None

```{autodoc2-docstring} src.interfaces.env.IEnv.name
```

````

````{py:property} device
:canonical: src.interfaces.env.IEnv.device
:type: torch.device

```{autodoc2-docstring} src.interfaces.env.IEnv.device
```

````

````{py:attribute} generator
:canonical: src.interfaces.env.IEnv.generator
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.interfaces.env.IEnv.generator
```

````

````{py:method} reset(tensordict: typing.Optional[tensordict.TensorDictBase] = None, **kwargs: typing.Any) -> tensordict.TensorDictBase
:canonical: src.interfaces.env.IEnv.reset

```{autodoc2-docstring} src.interfaces.env.IEnv.reset
```

````

````{py:method} step(tensordict: tensordict.TensorDictBase) -> tensordict.TensorDictBase
:canonical: src.interfaces.env.IEnv.step

```{autodoc2-docstring} src.interfaces.env.IEnv.step
```

````

````{py:method} get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.interfaces.env.IEnv.get_reward

```{autodoc2-docstring} src.interfaces.env.IEnv.get_reward
```

````

````{py:method} get_action_mask(tensordict: tensordict.TensorDictBase) -> torch.Tensor
:canonical: src.interfaces.env.IEnv.get_action_mask

```{autodoc2-docstring} src.interfaces.env.IEnv.get_action_mask
```

````

`````
