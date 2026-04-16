# {py:mod}`src.configs.policies.ph`

```{py:module} src.configs.policies.ph
```

```{autodoc2-docstring} src.configs.policies.ph
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PHConfig <src.configs.policies.ph.PHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ph.PHConfig
    :summary:
    ```
````

### API

`````{py:class} PHConfig
:canonical: src.configs.policies.ph.PHConfig

```{autodoc2-docstring} src.configs.policies.ph.PHConfig
```

````{py:attribute} rho
:canonical: src.configs.policies.ph.PHConfig.rho
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.rho
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ph.PHConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.max_iterations
```

````

````{py:attribute} convergence_tol
:canonical: src.configs.policies.ph.PHConfig.convergence_tol
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.convergence_tol
```

````

````{py:attribute} sub_solver
:canonical: src.configs.policies.ph.PHConfig.sub_solver
:type: str
:value: >
   'bc'

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.sub_solver
```

````

````{py:attribute} num_scenarios
:canonical: src.configs.policies.ph.PHConfig.num_scenarios
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.num_scenarios
```

````

````{py:attribute} horizon
:canonical: src.configs.policies.ph.PHConfig.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.horizon
```

````

````{py:attribute} consensus_scope
:canonical: src.configs.policies.ph.PHConfig.consensus_scope
:type: str
:value: >
   'day_0'

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.consensus_scope
```

````

````{py:attribute} stockout_penalty
:canonical: src.configs.policies.ph.PHConfig.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.stockout_penalty
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ph.PHConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.time_limit
```

````

````{py:attribute} verbose
:canonical: src.configs.policies.ph.PHConfig.verbose
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.verbose
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ph.PHConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ph.PHConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.route_improvement
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ph.PHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ph.PHConfig.seed
```

````

`````
