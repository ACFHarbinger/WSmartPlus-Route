# {py:mod}`src.interfaces.policy`

```{py:module} src.interfaces.policy
```

```{autodoc2-docstring} src.interfaces.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPolicy <src.interfaces.policy.IPolicy>`
  - ```{autodoc2-docstring} src.interfaces.policy.IPolicy
    :summary:
    ```
````

### API

`````{py:class} IPolicy
:canonical: src.interfaces.policy.IPolicy

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.policy.IPolicy
```

````{py:attribute} encoder
:canonical: src.interfaces.policy.IPolicy.encoder
:type: typing.Optional[torch.nn.Module]
:value: >
   None

```{autodoc2-docstring} src.interfaces.policy.IPolicy.encoder
```

````

````{py:attribute} decoder
:canonical: src.interfaces.policy.IPolicy.decoder
:type: typing.Optional[torch.nn.Module]
:value: >
   None

```{autodoc2-docstring} src.interfaces.policy.IPolicy.decoder
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[typing.Any] = None, strategy: str = 'sampling', num_starts: int = 1, **kwargs: typing.Any) -> typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]]
:canonical: src.interfaces.policy.IPolicy.forward

```{autodoc2-docstring} src.interfaces.policy.IPolicy.forward
```

````

````{py:method} __call__(*args, **kwargs) -> typing.Any
:canonical: src.interfaces.policy.IPolicy.__call__

```{autodoc2-docstring} src.interfaces.policy.IPolicy.__call__
```

````

`````
