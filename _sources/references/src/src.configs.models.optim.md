# {py:mod}`src.configs.models.optim`

```{py:module} src.configs.models.optim
```

```{autodoc2-docstring} src.configs.models.optim
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimConfig <src.configs.models.optim.OptimConfig>`
  - ```{autodoc2-docstring} src.configs.models.optim.OptimConfig
    :summary:
    ```
````

### API

`````{py:class} OptimConfig
:canonical: src.configs.models.optim.OptimConfig

```{autodoc2-docstring} src.configs.models.optim.OptimConfig
```

````{py:attribute} optimizer
:canonical: src.configs.models.optim.OptimConfig.optimizer
:type: str
:value: >
   'adam'

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.optimizer
```

````

````{py:attribute} lr
:canonical: src.configs.models.optim.OptimConfig.lr
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr
```

````

````{py:attribute} weight_decay
:canonical: src.configs.models.optim.OptimConfig.weight_decay
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.weight_decay
```

````

````{py:attribute} lr_scheduler
:canonical: src.configs.models.optim.OptimConfig.lr_scheduler
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_scheduler
```

````

````{py:attribute} lr_scheduler_kwargs
:canonical: src.configs.models.optim.OptimConfig.lr_scheduler_kwargs
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_scheduler_kwargs
```

````

````{py:attribute} lr_critic
:canonical: src.configs.models.optim.OptimConfig.lr_critic
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_critic
```

````

````{py:attribute} lr_decay
:canonical: src.configs.models.optim.OptimConfig.lr_decay
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_decay
```

````

````{py:attribute} lr_min_value
:canonical: src.configs.models.optim.OptimConfig.lr_min_value
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_min_value
```

````

````{py:attribute} lr_min_decay
:canonical: src.configs.models.optim.OptimConfig.lr_min_decay
:type: float
:value: >
   1e-08

```{autodoc2-docstring} src.configs.models.optim.OptimConfig.lr_min_decay
```

````

`````
