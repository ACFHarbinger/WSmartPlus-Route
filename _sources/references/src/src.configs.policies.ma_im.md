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

````{py:attribute} engine
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.engine
:type: str
:value: >
   'ma_im'

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.engine
```

````

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

````{py:attribute} must_go
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig.post_processing
```

````

`````
