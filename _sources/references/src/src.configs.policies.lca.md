# {py:mod}`src.configs.policies.lca`

```{py:module} src.configs.policies.lca
```

```{autodoc2-docstring} src.configs.policies.lca
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LCAConfig <src.configs.policies.lca.LCAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lca.LCAConfig
    :summary:
    ```
````

### API

`````{py:class} LCAConfig
:canonical: src.configs.policies.lca.LCAConfig

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.lca.LCAConfig.engine
:type: str
:value: >
   'lca'

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.engine
```

````

````{py:attribute} n_teams
:canonical: src.configs.policies.lca.LCAConfig.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.lca.LCAConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.max_iterations
```

````

````{py:attribute} tolerance_pct
:canonical: src.configs.policies.lca.LCAConfig.tolerance_pct
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.tolerance_pct
```

````

````{py:attribute} crossover_prob
:canonical: src.configs.policies.lca.LCAConfig.crossover_prob
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.crossover_prob
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.lca.LCAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.lca.LCAConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.lca.LCAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lca.LCAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lca.LCAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.lca.LCAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.lca.LCAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.lca.LCAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lca.LCAConfig.post_processing
```

````

`````
