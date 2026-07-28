# {py:mod}`src.configs.policies.tsp`

```{py:module} src.configs.policies.tsp
```

```{autodoc2-docstring} src.configs.policies.tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPConfig <src.configs.policies.tsp.TSPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig
    :summary:
    ```
````

### API

`````{py:class} TSPConfig
:canonical: src.configs.policies.tsp.TSPConfig

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig
```

````{py:attribute} cache
:canonical: src.configs.policies.tsp.TSPConfig.cache
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.cache
```

````

````{py:attribute} engine
:canonical: src.configs.policies.tsp.TSPConfig.engine
:type: str
:value: >
   'fast_tsp'

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.engine
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.tsp.TSPConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.tsp.TSPConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.tsp.TSPConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.tsp.TSPConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.tsp.TSPConfig.route_improvement
```

````

`````
