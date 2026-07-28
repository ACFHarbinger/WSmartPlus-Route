# {py:mod}`src.configs.policies.ma_im`

```{py:module} src.configs.policies.ma_im
```

```{autodoc2-docstring} src.configs.policies.ma_im
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmIslandModelConfig <src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmIslandModelConfig
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig
```

````{py:attribute} n_islands
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.n_islands
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.n_islands
```

````

````{py:attribute} island_size
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.island_size
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.island_size
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.max_generations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.max_generations
```

````

````{py:attribute} stagnation_limit
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.stagnation_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.stagnation_limit
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.n_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.route_improvement
```

````

`````
