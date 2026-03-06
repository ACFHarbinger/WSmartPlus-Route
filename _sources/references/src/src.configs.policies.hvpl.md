# {py:mod}`src.configs.policies.hvpl`

```{py:module} src.configs.policies.hvpl
```

```{autodoc2-docstring} src.configs.policies.hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HVPLConfig <src.configs.policies.hvpl.HVPLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig
    :summary:
    ```
````

### API

`````{py:class} HVPLConfig
:canonical: src.configs.policies.hvpl.HVPLConfig

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.hvpl.HVPLConfig.engine
:type: str
:value: >
   'hvpl'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.engine
```

````

````{py:attribute} n_teams
:canonical: src.configs.policies.hvpl.HVPLConfig.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.hvpl.HVPLConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.configs.policies.hvpl.HVPLConfig.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.sub_rate
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hvpl.HVPLConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hvpl.HVPLConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.seed
```

````

````{py:attribute} aco
:canonical: src.configs.policies.hvpl.HVPLConfig.aco
:type: src.configs.policies.aco.ACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.aco
```

````

````{py:attribute} alns
:canonical: src.configs.policies.hvpl.HVPLConfig.alns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.alns
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hvpl.HVPLConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hvpl.HVPLConfig.must_go
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hvpl.HVPLConfig.post_processing
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.post_processing
```

````

`````
