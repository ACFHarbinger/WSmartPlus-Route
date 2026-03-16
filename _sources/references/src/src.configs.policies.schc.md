# {py:mod}`src.configs.policies.schc`

```{py:module} src.configs.policies.schc
```

```{autodoc2-docstring} src.configs.policies.schc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCHCConfig <src.configs.policies.schc.SCHCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig
    :summary:
    ```
````

### API

`````{py:class} SCHCConfig
:canonical: src.configs.policies.schc.SCHCConfig

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.schc.SCHCConfig.engine
:type: str
:value: >
   'schc'

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.schc.SCHCConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.max_iterations
```

````

````{py:attribute} step_size
:canonical: src.configs.policies.schc.SCHCConfig.step_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.step_size
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.schc.SCHCConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.schc.SCHCConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.schc.SCHCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.schc.SCHCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.schc.SCHCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.schc.SCHCConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.schc.SCHCConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.schc.SCHCConfig.post_processing
```

````

`````
