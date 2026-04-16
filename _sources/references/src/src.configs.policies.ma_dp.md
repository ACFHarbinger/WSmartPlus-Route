# {py:mod}`src.configs.policies.ma_dp`

```{py:module} src.configs.policies.ma_dp
```

```{autodoc2-docstring} src.configs.policies.ma_dp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmDualPopulationConfig <src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmDualPopulationConfig
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig
```

````{py:attribute} population_size
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.max_iterations
```

````

````{py:attribute} diversity_injection_rate
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.diversity_injection_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.diversity_injection_rate
```

````

````{py:attribute} elite_learning_weights
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.elite_learning_weights
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.elite_learning_weights
```

````

````{py:attribute} elite_count
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.elite_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.elite_count
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.mandatory_selection
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.route_improvement
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ma_dp.MemeticAlgorithmDualPopulationConfig.route_improvement
```

````

`````
