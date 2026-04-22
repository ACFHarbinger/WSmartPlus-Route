# {py:mod}`src.configs.policies.lb_vns`

```{py:module} src.configs.policies.lb_vns
```

```{autodoc2-docstring} src.configs.policies.lb_vns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalBranchingVNSConfig <src.configs.policies.lb_vns.LocalBranchingVNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig
    :summary:
    ```
````

### API

`````{py:class} LocalBranchingVNSConfig
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.time_limit
```

````

````{py:attribute} k_min
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_min
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_min
```

````

````{py:attribute} k_max
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_max
:type: int
:value: >
   60

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_max
```

````

````{py:attribute} k_step
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_step
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.k_step
```

````

````{py:attribute} time_limit_per_lb
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.time_limit_per_lb
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.time_limit_per_lb
```

````

````{py:attribute} max_lb_iterations
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.max_lb_iterations
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.max_lb_iterations
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.mip_gap
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.vrpp
```

````

````{py:attribute} engine
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.engine
```

````

````{py:attribute} framework
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.framework
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.route_improvement
```

````

````{py:attribute} acceptance_criterion
:canonical: src.configs.policies.lb_vns.LocalBranchingVNSConfig.acceptance_criterion
:type: logic.src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lb_vns.LocalBranchingVNSConfig.acceptance_criterion
```

````

`````
