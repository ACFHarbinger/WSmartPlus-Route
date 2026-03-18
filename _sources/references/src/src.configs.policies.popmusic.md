# {py:mod}`src.configs.policies.popmusic`

```{py:module} src.configs.policies.popmusic
```

```{autodoc2-docstring} src.configs.policies.popmusic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POPMUSICSubSolverConfig <src.configs.policies.popmusic.POPMUSICSubSolverConfig>`
  - ```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICSubSolverConfig
    :summary:
    ```
* - {py:obj}`POPMUSICConfig <src.configs.policies.popmusic.POPMUSICConfig>`
  - ```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig
    :summary:
    ```
````

### API

`````{py:class} POPMUSICSubSolverConfig
:canonical: src.configs.policies.popmusic.POPMUSICSubSolverConfig

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICSubSolverConfig
```

````{py:attribute} fast_tsp
:canonical: src.configs.policies.popmusic.POPMUSICSubSolverConfig.fast_tsp
:type: typing.Optional[src.configs.policies.tsp.TSPConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICSubSolverConfig.fast_tsp
```

````

````{py:attribute} alns
:canonical: src.configs.policies.popmusic.POPMUSICSubSolverConfig.alns
:type: typing.Optional[src.configs.policies.alns.ALNSConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICSubSolverConfig.alns
```

````

````{py:attribute} hgs
:canonical: src.configs.policies.popmusic.POPMUSICSubSolverConfig.hgs
:type: typing.Optional[src.configs.policies.hgs.HGSConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICSubSolverConfig.hgs
```

````

`````

`````{py:class} POPMUSICConfig
:canonical: src.configs.policies.popmusic.POPMUSICConfig

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig
```

````{py:attribute} subproblem_size
:canonical: src.configs.policies.popmusic.POPMUSICConfig.subproblem_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.subproblem_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.popmusic.POPMUSICConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.max_iterations
```

````

````{py:attribute} base_solver
:canonical: src.configs.policies.popmusic.POPMUSICConfig.base_solver
:type: str
:value: >
   'alns'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.base_solver
```

````

````{py:attribute} base_solver_config
:canonical: src.configs.policies.popmusic.POPMUSICConfig.base_solver_config
:type: typing.Optional[src.configs.policies.popmusic.POPMUSICSubSolverConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.base_solver_config
```

````

````{py:attribute} cluster_solver
:canonical: src.configs.policies.popmusic.POPMUSICConfig.cluster_solver
:type: str
:value: >
   'fast_tsp'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.cluster_solver
```

````

````{py:attribute} cluster_solver_config
:canonical: src.configs.policies.popmusic.POPMUSICConfig.cluster_solver_config
:type: typing.Optional[src.configs.policies.popmusic.POPMUSICSubSolverConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.cluster_solver_config
```

````

````{py:attribute} initial_solver
:canonical: src.configs.policies.popmusic.POPMUSICConfig.initial_solver
:type: str
:value: >
   'nearest_neighbor'

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.initial_solver
```

````

````{py:attribute} seed
:canonical: src.configs.policies.popmusic.POPMUSICConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.popmusic.POPMUSICConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.popmusic.POPMUSICConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.popmusic.POPMUSICConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.popmusic.POPMUSICConfig.post_processing
```

````

`````
