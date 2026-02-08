# {py:mod}`src.configs.rl.imitation`

```{py:module} src.configs.rl.imitation
```

```{autodoc2-docstring} src.configs.rl.imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationConfig <src.configs.rl.imitation.ImitationConfig>`
  - ```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig
    :summary:
    ```
````

### API

`````{py:class} ImitationConfig
:canonical: src.configs.rl.imitation.ImitationConfig

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig
```

````{py:attribute} mode
:canonical: src.configs.rl.imitation.ImitationConfig.mode
:type: str
:value: >
   'hgs'

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig.mode
```

````

````{py:attribute} random_ls_iterations
:canonical: src.configs.rl.imitation.ImitationConfig.random_ls_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig.random_ls_iterations
```

````

````{py:attribute} random_ls_op_probs
:canonical: src.configs.rl.imitation.ImitationConfig.random_ls_op_probs
:type: typing.Optional[typing.Dict[str, float]]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig.random_ls_op_probs
```

````

````{py:attribute} enabled
:canonical: src.configs.rl.imitation.ImitationConfig.enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig.enabled
```

````

````{py:attribute} loss_fn
:canonical: src.configs.rl.imitation.ImitationConfig.loss_fn
:type: str
:value: >
   'nll'

```{autodoc2-docstring} src.configs.rl.imitation.ImitationConfig.loss_fn
```

````

`````
