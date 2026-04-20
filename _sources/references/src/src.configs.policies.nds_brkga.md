# {py:mod}`src.configs.policies.nds_brkga`

```{py:module} src.configs.policies.nds_brkga
```

```{autodoc2-docstring} src.configs.policies.nds_brkga
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDSBRKGAConfig <src.configs.policies.nds_brkga.NDSBRKGAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig
    :summary:
    ```
````

### API

`````{py:class} NDSBRKGAConfig
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.pop_size
:type: int
:value: >
   60

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.pop_size
```

````

````{py:attribute} n_elite
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.n_elite
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.n_elite
```

````

````{py:attribute} n_mutants
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.n_mutants
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.n_mutants
```

````

````{py:attribute} bias_elite
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.bias_elite
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.bias_elite
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.max_generations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.max_generations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.time_limit
:type: float
:value: >
   90.0

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.vrpp
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.overflow_penalty
```

````

````{py:attribute} seed_selection_strategy
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.seed_selection_strategy
:type: str
:value: >
   'fractional_knapsack'

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.seed_selection_strategy
```

````

````{py:attribute} seed_routing_strategy
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.seed_routing_strategy
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.seed_routing_strategy
```

````

````{py:attribute} n_seed_solutions
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.n_seed_solutions
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.n_seed_solutions
```

````

````{py:attribute} selection_threshold_min
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.selection_threshold_min
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.selection_threshold_min
```

````

````{py:attribute} selection_threshold_max
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.selection_threshold_max
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.selection_threshold_max
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.nds_brkga.NDSBRKGAConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.nds_brkga.NDSBRKGAConfig.route_improvement
```

````

`````
