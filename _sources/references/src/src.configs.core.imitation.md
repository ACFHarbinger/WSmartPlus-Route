# {py:mod}`src.configs.core.imitation`

```{py:module} src.configs.core.imitation
```

```{autodoc2-docstring} src.configs.core.imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationConfig <src.configs.core.imitation.ImitationConfig>`
  - ```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig
    :summary:
    ```
````

### API

`````{py:class} ImitationConfig
:canonical: src.configs.core.imitation.ImitationConfig

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig
```

````{py:attribute} mode
:canonical: src.configs.core.imitation.ImitationConfig.mode
:type: str
:value: >
   'hgs'

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig.mode
```

````

````{py:attribute} random_ls_iterations
:canonical: src.configs.core.imitation.ImitationConfig.random_ls_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig.random_ls_iterations
```

````

````{py:attribute} random_ls_op_probs
:canonical: src.configs.core.imitation.ImitationConfig.random_ls_op_probs
:type: typing.Optional[typing.Dict[str, float]]
:value: >
   None

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig.random_ls_op_probs
```

````

````{py:attribute} enabled
:canonical: src.configs.core.imitation.ImitationConfig.enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig.enabled
```

````

````{py:attribute} loss_fn
:canonical: src.configs.core.imitation.ImitationConfig.loss_fn
:type: str
:value: >
   'nll'

```{autodoc2-docstring} src.configs.core.imitation.ImitationConfig.loss_fn
```

````

`````
