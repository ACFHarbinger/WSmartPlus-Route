# {py:mod}`src.configs.policies.rrt`

```{py:module} src.configs.policies.rrt
```

```{autodoc2-docstring} src.configs.policies.rrt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RRTConfig <src.configs.policies.rrt.RRTConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig
    :summary:
    ```
````

### API

`````{py:class} RRTConfig
:canonical: src.configs.policies.rrt.RRTConfig

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.rrt.RRTConfig.engine
:type: str
:value: >
   'rr'

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.engine
```

````

````{py:attribute} tolerance
:canonical: src.configs.policies.rrt.RRTConfig.tolerance
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.tolerance
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.rrt.RRTConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.rrt.RRTConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.rrt.RRTConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.rrt.RRTConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rrt.RRTConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rrt.RRTConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.rrt.RRTConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.rrt.RRTConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rrt.RRTConfig.post_processing
```

````

`````
