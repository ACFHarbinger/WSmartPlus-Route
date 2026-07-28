# {py:mod}`src.configs.rl.policies.ils`

```{py:module} src.configs.rl.policies.ils
```

```{autodoc2-docstring} src.configs.rl.policies.ils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSConfig <src.configs.rl.policies.ils.ILSConfig>`
  - ```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig
    :summary:
    ```
````

### API

`````{py:class} ILSConfig
:canonical: src.configs.rl.policies.ils.ILSConfig

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig
```

````{py:attribute} n_restarts
:canonical: src.configs.rl.policies.ils.ILSConfig.n_restarts
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.n_restarts
```

````

````{py:attribute} ls_iterations
:canonical: src.configs.rl.policies.ils.ILSConfig.ls_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.ls_iterations
```

````

````{py:attribute} perturbation_strength
:canonical: src.configs.rl.policies.ils.ILSConfig.perturbation_strength
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.perturbation_strength
```

````

````{py:attribute} ls_operator
:canonical: src.configs.rl.policies.ils.ILSConfig.ls_operator
:type: typing.Union[str, typing.Dict[str, float]]
:value: >
   'two_opt'

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.ls_operator
```

````

````{py:attribute} perturbation_type
:canonical: src.configs.rl.policies.ils.ILSConfig.perturbation_type
:type: typing.Union[str, typing.Dict[str, float]]
:value: >
   'double_bridge'

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.perturbation_type
```

````

````{py:attribute} time_limit
:canonical: src.configs.rl.policies.ils.ILSConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.time_limit
```

````

````{py:attribute} op_probs
:canonical: src.configs.rl.policies.ils.ILSConfig.op_probs
:type: typing.Dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.op_probs
```

````

````{py:attribute} perturb_probs
:canonical: src.configs.rl.policies.ils.ILSConfig.perturb_probs
:type: typing.Dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.policies.ils.ILSConfig.perturb_probs
```

````

`````
