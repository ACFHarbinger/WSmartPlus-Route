# {py:mod}`src.configs.policies.cp_sat`

```{py:module} src.configs.policies.cp_sat
```

```{autodoc2-docstring} src.configs.policies.cp_sat
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CPSATConfig <src.configs.policies.cp_sat.CPSATConfig>`
  - ```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig
    :summary:
    ```
````

### API

`````{py:class} CPSATConfig
:canonical: src.configs.policies.cp_sat.CPSATConfig

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig
```

````{py:attribute} num_days
:canonical: src.configs.policies.cp_sat.CPSATConfig.num_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.num_days
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.cp_sat.CPSATConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.time_limit
```

````

````{py:attribute} search_workers
:canonical: src.configs.policies.cp_sat.CPSATConfig.search_workers
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.search_workers
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.cp_sat.CPSATConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.mip_gap
```

````

````{py:attribute} scaling_factor
:canonical: src.configs.policies.cp_sat.CPSATConfig.scaling_factor
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.scaling_factor
```

````

````{py:attribute} waste_weight
:canonical: src.configs.policies.cp_sat.CPSATConfig.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.configs.policies.cp_sat.CPSATConfig.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.cp_sat.CPSATConfig.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.overflow_penalty
```

````

````{py:attribute} mean_increment
:canonical: src.configs.policies.cp_sat.CPSATConfig.mean_increment
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.mean_increment
```

````

````{py:attribute} seed
:canonical: src.configs.policies.cp_sat.CPSATConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.cp_sat.CPSATConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.cp_sat.CPSATConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cp_sat.CPSATConfig.post_processing
```

````

`````
