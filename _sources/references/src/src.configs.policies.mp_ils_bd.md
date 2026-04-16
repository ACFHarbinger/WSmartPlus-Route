# {py:mod}`src.configs.policies.mp_ils_bd`

```{py:module} src.configs.policies.mp_ils_bd
```

```{autodoc2-docstring} src.configs.policies.mp_ils_bd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MPILSBDConfig <src.configs.policies.mp_ils_bd.MPILSBDConfig>`
  - ```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig
    :summary:
    ```
````

### API

`````{py:class} MPILSBDConfig
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.time_limit
```

````

````{py:attribute} master_time_limit
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.master_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.master_time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.mip_gap
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.max_iterations
```

````

````{py:attribute} theta_lower_bound
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.theta_lower_bound
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.theta_lower_bound
```

````

````{py:attribute} max_cuts_per_round
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.max_cuts_per_round
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.max_cuts_per_round
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.enable_comb_cuts
```

````

````{py:attribute} verbose
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.verbose
```

````

````{py:attribute} horizon
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.horizon
```

````

````{py:attribute} stockout_penalty
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.stockout_penalty
```

````

````{py:attribute} big_m
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.big_m
:type: float
:value: >
   10000.0

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.big_m
```

````

````{py:attribute} mean_scenario_only
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.mean_scenario_only
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.mean_scenario_only
```

````

````{py:attribute} initial_inventory
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.initial_inventory
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.initial_inventory
```

````

````{py:attribute} seed
:canonical: src.configs.policies.mp_ils_bd.MPILSBDConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_ils_bd.MPILSBDConfig.seed
```

````

`````
