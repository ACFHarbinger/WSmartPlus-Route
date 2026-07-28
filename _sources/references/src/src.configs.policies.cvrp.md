# {py:mod}`src.configs.policies.cvrp`

```{py:module} src.configs.policies.cvrp
```

```{autodoc2-docstring} src.configs.policies.cvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPConfig <src.configs.policies.cvrp.CVRPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig
    :summary:
    ```
````

### API

`````{py:class} CVRPConfig
:canonical: src.configs.policies.cvrp.CVRPConfig

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig
```

````{py:attribute} cache
:canonical: src.configs.policies.cvrp.CVRPConfig.cache
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.cache
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.cvrp.CVRPConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.time_limit
```

````

````{py:attribute} engine
:canonical: src.configs.policies.cvrp.CVRPConfig.engine
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.engine
```

````

````{py:attribute} seed
:canonical: src.configs.policies.cvrp.CVRPConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.cvrp.CVRPConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.cvrp.CVRPConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cvrp.CVRPConfig.route_improvement
```

````

`````
