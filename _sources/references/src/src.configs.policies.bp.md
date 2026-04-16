# {py:mod}`src.configs.policies.bp`

```{py:module} src.configs.policies.bp
```

```{autodoc2-docstring} src.configs.policies.bp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPConfig <src.configs.policies.bp.BPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bp.BPConfig
    :summary:
    ```
````

### API

`````{py:class} BPConfig
:canonical: src.configs.policies.bp.BPConfig

```{autodoc2-docstring} src.configs.policies.bp.BPConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.bp.BPConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.bp.BPConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.max_iterations
```

````

````{py:attribute} max_routes_per_iteration
:canonical: src.configs.policies.bp.BPConfig.max_routes_per_iteration
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.max_routes_per_iteration
```

````

````{py:attribute} optimality_gap
:canonical: src.configs.policies.bp.BPConfig.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.optimality_gap
```

````

````{py:attribute} use_ryan_foster_branching
:canonical: src.configs.policies.bp.BPConfig.use_ryan_foster_branching
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.use_ryan_foster_branching
```

````

````{py:attribute} multi_day_mode
:canonical: src.configs.policies.bp.BPConfig.multi_day_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.multi_day_mode
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.bp.BPConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.bp.BPConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.route_improvement
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bp.BPConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bp.BPConfig.seed
```

````

`````
