# {py:mod}`src.configs.policies.lbbd`

```{py:module} src.configs.policies.lbbd
```

```{autodoc2-docstring} src.configs.policies.lbbd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBBDConfig <src.configs.policies.lbbd.LBBDConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig
    :summary:
    ```
````

### API

`````{py:class} LBBDConfig
:canonical: src.configs.policies.lbbd.LBBDConfig

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig
```

````{py:attribute} num_days
:canonical: src.configs.policies.lbbd.LBBDConfig.num_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.num_days
```

````

````{py:attribute} stochastic_master
:canonical: src.configs.policies.lbbd.LBBDConfig.stochastic_master
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.stochastic_master
```

````

````{py:attribute} mean_increment
:canonical: src.configs.policies.lbbd.LBBDConfig.mean_increment
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.mean_increment
```

````

````{py:attribute} num_scenarios
:canonical: src.configs.policies.lbbd.LBBDConfig.num_scenarios
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.num_scenarios
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.lbbd.LBBDConfig.max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.max_iterations
```

````

````{py:attribute} benders_gap
:canonical: src.configs.policies.lbbd.LBBDConfig.benders_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.benders_gap
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.lbbd.LBBDConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.time_limit
```

````

````{py:attribute} subproblem_timeout
:canonical: src.configs.policies.lbbd.LBBDConfig.subproblem_timeout
:type: float
:value: >
   20.0

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.subproblem_timeout
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.lbbd.LBBDConfig.mip_gap
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.mip_gap
```

````

````{py:attribute} waste_weight
:canonical: src.configs.policies.lbbd.LBBDConfig.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.configs.policies.lbbd.LBBDConfig.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.lbbd.LBBDConfig.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.overflow_penalty
```

````

````{py:attribute} use_nogood_cuts
:canonical: src.configs.policies.lbbd.LBBDConfig.use_nogood_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.use_nogood_cuts
```

````

````{py:attribute} use_optimality_cuts
:canonical: src.configs.policies.lbbd.LBBDConfig.use_optimality_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.use_optimality_cuts
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lbbd.LBBDConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.seed
```

````

````{py:method} __post_init__()
:canonical: src.configs.policies.lbbd.LBBDConfig.__post_init__

```{autodoc2-docstring} src.configs.policies.lbbd.LBBDConfig.__post_init__
```

````

`````
