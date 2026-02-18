# {py:mod}`src.configs.policies.ahvpl`

```{py:module} src.configs.policies.ahvpl
```

```{autodoc2-docstring} src.configs.policies.ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AHVPLConfig <src.configs.policies.ahvpl.AHVPLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig
    :summary:
    ```
````

### API

`````{py:class} AHVPLConfig
:canonical: src.configs.policies.ahvpl.AHVPLConfig

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.ahvpl.AHVPLConfig.engine
:type: str
:value: >
   'ahvpl'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.engine
```

````

````{py:attribute} n_teams
:canonical: src.configs.policies.ahvpl.AHVPLConfig.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ahvpl.AHVPLConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.configs.policies.ahvpl.AHVPLConfig.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.sub_rate
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ahvpl.AHVPLConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.time_limit
```

````

````{py:attribute} hgs
:canonical: src.configs.policies.ahvpl.AHVPLConfig.hgs
:type: src.configs.policies.hgs.HGSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.hgs
```

````

````{py:attribute} aco
:canonical: src.configs.policies.ahvpl.AHVPLConfig.aco
:type: src.configs.policies.aco.ACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.aco
```

````

````{py:attribute} alns
:canonical: src.configs.policies.ahvpl.AHVPLConfig.alns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.alns
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ahvpl.AHVPLConfig.must_go
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ahvpl.AHVPLConfig.post_processing
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ahvpl.AHVPLConfig.post_processing
```

````

`````
