# {py:mod}`src.configs.policies.hils`

```{py:module} src.configs.policies.hils
```

```{autodoc2-docstring} src.configs.policies.hils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HILSConfig <src.configs.policies.hils.HILSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hils.HILSConfig
    :summary:
    ```
````

### API

`````{py:class} HILSConfig
:canonical: src.configs.policies.hils.HILSConfig

Bases: {py:obj}`src.configs.policies.abc.ABCConfig`

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig
```

````{py:attribute} max_iterations
:canonical: src.configs.policies.hils.HILSConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.max_iterations
```

````

````{py:attribute} ils_iterations
:canonical: src.configs.policies.hils.HILSConfig.ils_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.ils_iterations
```

````

````{py:attribute} perturbation_size
:canonical: src.configs.policies.hils.HILSConfig.perturbation_size
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.perturbation_size
```

````

````{py:attribute} use_set_partitioning
:canonical: src.configs.policies.hils.HILSConfig.use_set_partitioning
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.use_set_partitioning
```

````

````{py:attribute} sp_time_limit
:canonical: src.configs.policies.hils.HILSConfig.sp_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.sp_time_limit
```

````

````{py:attribute} sp_mip_gap
:canonical: src.configs.policies.hils.HILSConfig.sp_mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.sp_mip_gap
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hils.HILSConfig.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hils.HILSConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.hils.HILSConfig.seed
```

````

`````
