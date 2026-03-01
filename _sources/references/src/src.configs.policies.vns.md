# {py:mod}`src.configs.policies.vns`

```{py:module} src.configs.policies.vns
```

```{autodoc2-docstring} src.configs.policies.vns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VNSConfig <src.configs.policies.vns.VNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.vns.VNSConfig
    :summary:
    ```
````

### API

`````{py:class} VNSConfig
:canonical: src.configs.policies.vns.VNSConfig

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.vns.VNSConfig.engine
:type: str
:value: >
   'vns'

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.engine
```

````

````{py:attribute} k_max
:canonical: src.configs.policies.vns.VNSConfig.k_max
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.k_max
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.vns.VNSConfig.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.vns.VNSConfig.local_search_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.local_search_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.vns.VNSConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.vns.VNSConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.vns.VNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.vns.VNSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.vns.VNSConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.vns.VNSConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.vns.VNSConfig.post_processing
```

````

`````
