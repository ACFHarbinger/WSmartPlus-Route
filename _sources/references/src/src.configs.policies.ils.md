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

````{py:attribute} vrpp
:canonical: src.configs.policies.ils.ILSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ils.ILSConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ils.ILSConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ils.ILSConfig.post_processing
```

````

`````
