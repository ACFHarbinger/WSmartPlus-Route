# {py:mod}`src.configs.policies.ils_bd`

```{py:module} src.configs.policies.ils_bd
```

```{autodoc2-docstring} src.configs.policies.ils_bd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IntegerLShapedBendersConfig <src.configs.policies.ils_bd.IntegerLShapedBendersConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig
    :summary:
    ```
````

### API

`````{py:class} IntegerLShapedBendersConfig
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.time_limit
```

````

````{py:attribute} master_time_limit
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.master_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.master_time_limit
```

````

````{py:attribute} n_scenarios
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.n_scenarios
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.n_scenarios
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.profit_aware_operators
```

````

````{py:attribute} max_benders_iterations
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.max_benders_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.max_benders_iterations
```

````

````{py:attribute} benders_gap
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.benders_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.benders_gap
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.overflow_penalty
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.overflow_penalty
```

````

````{py:attribute} undervisit_penalty
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.undervisit_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.undervisit_penalty
```

````

````{py:attribute} collection_threshold
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.collection_threshold
:type: float
:value: >
   70.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.collection_threshold
```

````

````{py:attribute} horizon
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.horizon
```

````

````{py:attribute} stockout_penalty
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.stockout_penalty
```

````

````{py:attribute} big_m
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.big_m
:type: float
:value: >
   10000.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.big_m
```

````

````{py:attribute} mean_scenario_only
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mean_scenario_only
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mean_scenario_only
```

````

````{py:attribute} initial_inventory
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.initial_inventory
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.initial_inventory
```

````

````{py:attribute} fill_rate_cv
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.fill_rate_cv
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.fill_rate_cv
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mip_gap
```

````

````{py:attribute} theta_lower_bound
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.theta_lower_bound
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.theta_lower_bound
```

````

````{py:attribute} verbose
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.verbose
```

````

````{py:attribute} max_cuts_per_round
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.max_cuts_per_round
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.max_cuts_per_round
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.enable_comb_cuts
```

````

````{py:attribute} engine
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.engine
```

````

````{py:attribute} framework
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.framework
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ils_bd.IntegerLShapedBendersConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ils_bd.IntegerLShapedBendersConfig.route_improvement
```

````

`````
