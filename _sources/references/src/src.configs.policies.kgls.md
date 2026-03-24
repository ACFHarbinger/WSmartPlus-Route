# {py:mod}`src.configs.policies.kgls`

```{py:module} src.configs.policies.kgls
```

```{autodoc2-docstring} src.configs.policies.kgls
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KGLSConfig <src.configs.policies.kgls.KGLSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig
    :summary:
    ```
````

### API

`````{py:class} KGLSConfig
:canonical: src.configs.policies.kgls.KGLSConfig

Bases: {py:obj}`src.configs.policies.abc.ABCConfig`

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.kgls.KGLSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.time_limit
```

````

````{py:attribute} num_perturbations
:canonical: src.configs.policies.kgls.KGLSConfig.num_perturbations
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.num_perturbations
```

````

````{py:attribute} neighborhood_size
:canonical: src.configs.policies.kgls.KGLSConfig.neighborhood_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.neighborhood_size
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.kgls.KGLSConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.local_search_iterations
```

````

````{py:attribute} moves
:canonical: src.configs.policies.kgls.KGLSConfig.moves
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.moves
```

````

````{py:attribute} penalization_cycle
:canonical: src.configs.policies.kgls.KGLSConfig.penalization_cycle
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.penalization_cycle
```

````

````{py:attribute} seed
:canonical: src.configs.policies.kgls.KGLSConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.kgls.KGLSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.kgls.KGLSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.kgls.KGLSConfig.profit_aware_operators
```

````

`````
