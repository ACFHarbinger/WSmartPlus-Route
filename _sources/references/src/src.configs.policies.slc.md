# {py:mod}`src.configs.policies.slc`

```{py:module} src.configs.policies.slc
```

```{autodoc2-docstring} src.configs.policies.slc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SLCConfig <src.configs.policies.slc.SLCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.slc.SLCConfig
    :summary:
    ```
````

### API

`````{py:class} SLCConfig
:canonical: src.configs.policies.slc.SLCConfig

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.slc.SLCConfig.engine
:type: str
:value: >
   'slc'

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.engine
```

````

````{py:attribute} n_teams
:canonical: src.configs.policies.slc.SLCConfig.n_teams
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.n_teams
```

````

````{py:attribute} team_size
:canonical: src.configs.policies.slc.SLCConfig.team_size
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.team_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.slc.SLCConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.max_iterations
```

````

````{py:attribute} stagnation_limit
:canonical: src.configs.policies.slc.SLCConfig.stagnation_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.stagnation_limit
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.slc.SLCConfig.n_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.slc.SLCConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.slc.SLCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.slc.SLCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.slc.SLCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.slc.SLCConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.slc.SLCConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.slc.SLCConfig.post_processing
```

````

`````
