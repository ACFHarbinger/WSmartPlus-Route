# {py:mod}`src.configs.policies.lahc`

```{py:module} src.configs.policies.lahc
```

```{autodoc2-docstring} src.configs.policies.lahc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LAHCConfig <src.configs.policies.lahc.LAHCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig
    :summary:
    ```
````

### API

`````{py:class} LAHCConfig
:canonical: src.configs.policies.lahc.LAHCConfig

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.lahc.LAHCConfig.engine
:type: str
:value: >
   'lahc'

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.engine
```

````

````{py:attribute} queue_size
:canonical: src.configs.policies.lahc.LAHCConfig.queue_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.queue_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.lahc.LAHCConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.lahc.LAHCConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.lahc.LAHCConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.lahc.LAHCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lahc.LAHCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.lahc.LAHCConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lahc.LAHCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.lahc.LAHCConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.lahc.LAHCConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lahc.LAHCConfig.route_improvement
```

````

`````
