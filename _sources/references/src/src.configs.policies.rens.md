# {py:mod}`src.configs.policies.rens`

```{py:module} src.configs.policies.rens
```

```{autodoc2-docstring} src.configs.policies.rens
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RENSConfig <src.configs.policies.rens.RENSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rens.RENSConfig
    :summary:
    ```
````

### API

`````{py:class} RENSConfig
:canonical: src.configs.policies.rens.RENSConfig

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.rens.RENSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.time_limit
```

````

````{py:attribute} lp_time_limit
:canonical: src.configs.policies.rens.RENSConfig.lp_time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.lp_time_limit
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.configs.policies.rens.RENSConfig.mip_limit_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.mip_limit_nodes
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.rens.RENSConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.mip_gap
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rens.RENSConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rens.RENSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.vrpp
```

````

````{py:attribute} engine
:canonical: src.configs.policies.rens.RENSConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.rens.RENSConfig.must_go
:type: typing.Optional[src.configs.policies.other.must_go.MustGoConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.rens.RENSConfig.post_processing
:type: typing.Optional[src.configs.policies.other.post_processing.PostProcessingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rens.RENSConfig.post_processing
```

````

`````
