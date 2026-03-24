# {py:mod}`src.configs.policies.aco_hh`

```{py:module} src.configs.policies.aco_hh
```

```{autodoc2-docstring} src.configs.policies.aco_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperHeuristicACOConfig <src.configs.policies.aco_hh.HyperHeuristicACOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig
    :summary:
    ```
````

### API

`````{py:class} HyperHeuristicACOConfig
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig
```

````{py:attribute} n_ants
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.n_ants
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.n_ants
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.beta
```

````

````{py:attribute} rho
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.rho
```

````

````{py:attribute} tau_0
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.seed
```

````

````{py:attribute} q0
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.q0
```

````

````{py:attribute} sequence_length
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.sequence_length
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.sequence_length
```

````

````{py:attribute} local_search
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.local_search
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.local_search
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.local_search_iterations
```

````

````{py:attribute} elitist_weight
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.elitist_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.elitist_weight
```

````

````{py:attribute} operators
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.operators
```

````

````{py:attribute} engine
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.engine
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.aco_hh.HyperHeuristicACOConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aco_hh.HyperHeuristicACOConfig.post_processing
```

````

`````
