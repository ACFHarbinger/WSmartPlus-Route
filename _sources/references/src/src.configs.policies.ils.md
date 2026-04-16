# {py:mod}`src.configs.policies.ils`

```{py:module} src.configs.policies.ils
```

```{autodoc2-docstring} src.configs.policies.ils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSConfig <src.configs.policies.ils.ILSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ils.ILSConfig
    :summary:
    ```
````

### API

`````{py:class} ILSConfig
:canonical: src.configs.policies.ils.ILSConfig

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.ils.ILSConfig.engine
:type: str
:value: >
   'ils'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.engine
```

````

````{py:attribute} n_restarts
:canonical: src.configs.policies.ils.ILSConfig.n_restarts
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.n_restarts
```

````

````{py:attribute} inner_iterations
:canonical: src.configs.policies.ils.ILSConfig.inner_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.inner_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ils.ILSConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.ils.ILSConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.n_llh
```

````

````{py:attribute} perturbation_strength
:canonical: src.configs.policies.ils.ILSConfig.perturbation_strength
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.perturbation_strength
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ils.ILSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ils.ILSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ils.ILSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ils.ILSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ils.ILSConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ils.ILSConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.route_improvement
```

````

````{py:attribute} acceptance
:canonical: src.configs.policies.ils.ILSConfig.acceptance
:type: logic.src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.acceptance
```

````

`````
