# {py:mod}`src.configs.policies.arco`

```{py:module} src.configs.policies.arco
```

```{autodoc2-docstring} src.configs.policies.arco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ARCOConfig <src.configs.policies.arco.ARCOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig
    :summary:
    ```
````

### API

`````{py:class} ARCOConfig
:canonical: src.configs.policies.arco.ARCOConfig

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig
```

````{py:attribute} constructors
:canonical: src.configs.policies.arco.ARCOConfig.constructors
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.constructors
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.arco.ARCOConfig.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.time_limit
```

````

````{py:attribute} selection_strategy
:canonical: src.configs.policies.arco.ARCOConfig.selection_strategy
:type: str
:value: >
   'epsilon_greedy'

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.selection_strategy
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.arco.ARCOConfig.epsilon
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.epsilon
```

````

````{py:attribute} temperature
:canonical: src.configs.policies.arco.ARCOConfig.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.temperature
```

````

````{py:attribute} alpha_ema
:canonical: src.configs.policies.arco.ARCOConfig.alpha_ema
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.alpha_ema
```

````

````{py:attribute} weight_init
:canonical: src.configs.policies.arco.ARCOConfig.weight_init
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.weight_init
```

````

````{py:attribute} weight_floor
:canonical: src.configs.policies.arco.ARCOConfig.weight_floor
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.weight_floor
```

````

````{py:attribute} decay
:canonical: src.configs.policies.arco.ARCOConfig.decay
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.decay
```

````

````{py:attribute} seed
:canonical: src.configs.policies.arco.ARCOConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.arco.ARCOConfig.seed
```

````

`````
