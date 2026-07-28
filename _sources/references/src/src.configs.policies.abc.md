# {py:mod}`src.configs.policies.abc`

```{py:module} src.configs.policies.abc
```

```{autodoc2-docstring} src.configs.policies.abc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABCConfig <src.configs.policies.abc.ABCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.abc.ABCConfig
    :summary:
    ```
````

### API

`````{py:class} ABCConfig
:canonical: src.configs.policies.abc.ABCConfig

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig
```

````{py:attribute} n_sources
:canonical: src.configs.policies.abc.ABCConfig.n_sources
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.n_sources
```

````

````{py:attribute} limit
:canonical: src.configs.policies.abc.ABCConfig.limit
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.abc.ABCConfig.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.abc.ABCConfig.n_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.abc.ABCConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.abc.ABCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.abc.ABCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.abc.ABCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.abc.ABCConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.abc.ABCConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.abc.ABCConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.abc.ABCConfig.route_improvement
```

````

`````
